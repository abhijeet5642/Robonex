import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from fastapi import FastAPI
import uvicorn
import threading
import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==========================================
# LLM SETUP & MODELS
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Use AsyncGroq for FastAPI compatibility
client = AsyncGroq(api_key=GROQ_API_KEY)

class CommandRequest(BaseModel):
    text: str

class LLMCommand(BaseModel):
    x: float = Field(default=0.0, ge=-1.0, le=1.0) 
    y: float = Field(default=0.0, ge=-1.0, le=1.0)
    yaw: float = Field(default=0.0, ge=-1.0, le=1.0)
    stance: str = Field(default="stand")
    gait: str = Field(default="walk")
    speed: str = Field(default="slow")
    mode: str = Field(default="ACTIVE")

# ==========================================
# ROS 2 NODE (THE PUBLISHER)
# ==========================================
class AINode(Node):
    def __init__(self):
        super().__init__('ai_node')
        self.publisher_ = self.create_publisher(String, 'robot_commands', 10)
        self.get_logger().info("🚀 AI Node & FastAPI Server Started!")

    def publish_movement(self, state_dict):
        msg = String()
        msg.data = json.dumps(state_dict) 
        self.publisher_.publish(msg)
        self.get_logger().info(f" Published to ROS: {msg.data}")

# ==========================================
# FASTAPI SERVER
# ==========================================
app = FastAPI()
ros_node = None

# Global memory to track the last known state
current_robot_state = {
    "x": 0.0, "y": 0.0, "yaw": 0.0, 
    "speed": "slow", "stance": "stand", "gait": "walk", "mode": "ACTIVE"
}

@app.post("/command")
async def receive_command(request: CommandRequest):
    global current_robot_state
    text_command = request.text.strip()
    print(f" User Said: '{text_command}'")

    system_rules = """
You are a quadruped robot controller. Your job is to interpret natural conversational language and convert the user's intent into a specific JSON state. Only return raw JSON.

REQUIRED JSON KEYS:
"x" (float: -1.0 to 1.0), "y" (float: -1.0 to 1.0), "yaw" (float: -1.0 to 1.0),
"mode" ("ACTIVE" or "SLEEP"),
"stance" ("stand" or "sit"),
"gait" ("walk", "trot", "crawl"),
"speed" ("slow", "medium", "fast").

DIRECTION & AXIS MAPPING (-1.0 to 1.0):
x axis (forward/backward movement)
- "move forward", "go forward", "walk ahead" → x = 1.0
- "move backward", "go back" → x = -1.0

y axis (side movement / strafing)
- "move right", "strafe right" → y = 1.0
- "move left", "strafe left" → y = -1.0

yaw axis (rotation)
- "turn right", "rotate right" → yaw = 1.0
- "turn left", "rotate left" → yaw = -1.0

RULE 1: NEW MOVEMENT COMMANDS (STRICT RESET)
1. Completely RESET previous motion values.
2. Set ONLY the requested axis.
3. FORCE the other two axes to 0.0.
4. NEVER mix axes from previous history.

RULE 2: SPEED UP / FASTER (USING MEMORY)
- KEEP previous x, y, yaw.
- Increase speed: slow → medium → fast
- Update gait: slow=walk, medium=trot, fast=crawl

RULE 3: SLOW DOWN / SLOWER (USING MEMORY)
- KEEP previous x, y, yaw.
- Decrease speed: fast → medium → slow
- Update gait based on speed.

RULE 4: STOP COMMAND
"stop", "halt", "freeze", "wait" → x=0.0, y=0.0, yaw=0.0, speed="slow", gait="walk", stance="stand", mode="ACTIVE"

RULE 5: STANCE CONTROL
"sit", "rest" → stance="sit", x=0.0, y=0.0, yaw=0.0
"stand", "ready" → stance="stand"

RULE 6: UNKNOWN OR NON-ROBOT COMMANDS (IGNORE)
DO NOT reset movement. Keep previous robot state unchanged.
"""

    user_prompt = f"""
    LAST ROBOT STATE:
    - Direction: x={current_robot_state['x']}, y={current_robot_state['y']}, yaw={current_robot_state['yaw']}
    - Speed Level: {current_robot_state['speed']}
    
    NEW USER COMMAND: '{text_command}'
    """

    try:
        start_time = time.time() 
        
        # Make the request with the new parameters
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        
        # Accumulate the streamed chunks
        raw_message = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                raw_message += chunk.choices[0].delta.content
        
        latency = time.time() - start_time
        print(f" Model Latency: {latency:.2f} seconds")

        if not raw_message:
            raise ValueError("The LLM returned an empty response.")
            
        raw_content = raw_message.strip()
        
        # Clean up formatting if the LLM wraps the JSON in a markdown code block
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.replace("```", "").strip()

        # Parse and validate the JSON
        raw_dict = json.loads(raw_content)
        validated_command = LLMCommand(**raw_dict)
        
        current_robot_state = validated_command.model_dump()
        
        if ros_node:
            ros_node.publish_movement(current_robot_state)
            
        return {"status": "success", "data": current_robot_state}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"status": "error", "detail": str(e)}

def ros_spin_thread():
    rclpy.spin(ros_node)

def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = AINode()
    thread = threading.Thread(target=ros_spin_thread, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=5000)
    rclpy.shutdown()

if __name__ == '__main__':
    main()