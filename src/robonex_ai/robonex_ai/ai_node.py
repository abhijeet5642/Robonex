import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from fastapi import FastAPI
import uvicorn
import threading
from datetime import datetime, timezone, timedelta
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{severity}] {message}'

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
    mode: int = Field(default=1)
    x_velocity: float = Field(default=0.0, ge=-0.5, le=0.5)
    y_velocity: float = Field(default=0.0, ge=-0.4, le=0.4)
    yaw_velocity: float = Field(default=0.0, ge=-0.8, le=0.8)
    speed: str = Field(default="slow")  # <-- ADDED speed back


# ==========================================
# ROS 2 NODE (THE PUBLISHER)
# ==========================================
class AINode(Node):
    def __init__(self):
        super().__init__('ai_node')
        self.publisher_ = self.create_publisher(String, 'robot_commands', 10)
        self.get_logger().info("🚀 AI Node & FastAPI Server Started!")

    def publish_movement(self, state_dict):
        # 1. Prepare and send the REAL payload to ROS
        msg = String()
        msg.data = json.dumps(state_dict) 
        self.publisher_.publish(msg)
        
        # 2. Create a "display copy" for your terminal
        display_dict = state_dict.copy()
        mode_names = {0: "sleep", 1: "stand", 4: "move"}
        display_dict["mode"] = mode_names.get(display_dict["mode"], "unknown")
        
        # 3. Calculate Indian Standard Time (IST)
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        ist_time = datetime.now(ist_offset).strftime('%d-%m-%Y %I:%M:%S %p')
        
        # 4. Print the beautifully formatted message!
        self.get_logger().info(f"[{ist_time}] [ai_node]:\n Published to ROS: {json.dumps(display_dict)}")

# ==========================================
# FASTAPI SERVER
# ==========================================
app = FastAPI()
ros_node = None

# Global memory to track the last known state
current_robot_state = {
    "mode": 1,
    "x_velocity": 0.0,
    "y_velocity": 0.0,
    "yaw_velocity": 0.0,
    "speed": "slow",  
}

command_memory = []


@app.post("/command")
async def receive_command(request: CommandRequest):
    global current_robot_state, command_memory
    text_command = request.text.strip()
    print(f"\n User Said: '{text_command}'")

    system_rules = """You control a quadruped robot through voice commands and mode selection.

            The robot has three operational modes:
            - mode 0 (sleep): Robot is inactive/resting on the ground
            - mode 1 (fixed_stand): Robot stands in a fixed position
            - mode 4 (move): Robot accepts velocity commands for movement

            For movement commands (when mode should be 4), include these velocity values:
            - x_velocity: forward/backward (-0.5 to 0.5)
            - y_velocity: left/right (-0.4 to 0.4)
            - yaw_velocity: rotation (-0.8 to 0.8)

            COMMAND MEMORY INSTRUCTIONS:
            1. When the user refers to previous commands (e.g., "do that again", "faster", "slower", "repeat", "like before"), use the command history to determine what "that" refers to.
            2. If the user says "faster" or "slower", modify the velocity values from the most recent movement command.
            3. If the user refers to "again" or "repeat", use the same mode and velocities as the previous command.
            4. For contextual follow-ups (e.g., "now turn right", "stop", "keep going"), maintain context from the previous commands.
            5. If current command is ambiguous but previous commands provide context, use that context to interpret the command.

            MODE SELECTION INSTRUCTIONS:
            1. MODE SELECTION INSTRUCTIONS:
            1. For commands like "sleep", "rest", "lie down", "deactivate", "take a load off", "sit down", set mode to 0
            2. For commands like "stand up", "get up", "get ready", "rise", "stop", "pause", "halt", "stand still", set mode to 1
            3. For any movement command, set mode to 4 and include appropriate velocity values
            4. If a command doesn't clearly specify mode but implies movement, assume mode 4
            5. If current mode if 4, then switch only to mode 1 if the command is not movement related
            6. If current mode is 1, then switch only to mode 0 if the command is not movement related
            7. If current mode is unknown, interpret the command in the most appropriate mode

            Respond with JSON object containing:
            - "mode": integer (0, 1, or 4)
            - "x_velocity": float (-0.5 to 0.5), only when mode is 4
            - "y_velocity": float (-0.4 to 0.4), only when mode is 4
            - "yaw_velocity": float (-0.8 to 0.8), only when mode is 4
            - "speed": string ("slow", "medium", or "fast") based on the movement speed
           SPEED & DIRECTION MAPPING:
            Use these exact values based on the implied speed (default to "slow"):
            - "slow": x=0.2, y=0.15, yaw=0.3
            - "medium": x=0.35, y=0.25, yaw=0.5
            - "fast": x=0.5, y=0.4, yaw=0.8
            
            SIGN CONVENTIONS (CRITICAL):
            - X-Axis: Forward is POSITIVE (+), Backward is NEGATIVE (-)
            - Y-Axis: Left is POSITIVE (+), Right is NEGATIVE (-)
            - Yaw-Axis: Turning Left (counter-clockwise) is POSITIVE (+), Turning Right (clockwise) is NEGATIVE (-)
        """

    history_text = "None"
    if command_memory:
        history_text = "\n".join(
            [
                f"- User: '{cmd['text']}' -> Robot: {cmd['state']}"
                for cmd in command_memory
            ]
        )

    user_prompt = f"""
    PREVIOUS COMMAND HISTORY:
    {history_text}

    LAST ROBOT STATE:
    - Mode: {current_robot_state['mode']}
    - Velocities: x={current_robot_state['x_velocity']}, y={current_robot_state['y_velocity']}, yaw={current_robot_state['yaw_velocity']}
    - Speed: {current_robot_state['speed']}
    
    NEW USER COMMAND: '{text_command}'
    """

    try:
        start_time = time.time()

        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None,
        )

        raw_message = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                raw_message += chunk.choices[0].delta.content

        latency = time.time() - start_time
        print(f" Model Latency: {latency:.2f} seconds")

        if not raw_message:
            raise ValueError("The LLM returned an empty response.")

        raw_content = raw_message.strip()

        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.replace("```", "").strip()

        raw_dict = json.loads(raw_content)
        validated_command = LLMCommand(**raw_dict)

        current_robot_state = validated_command.model_dump()

        command_memory.append({"text": text_command, "state": current_robot_state})
        if len(command_memory) > 5:
            command_memory.pop(0)

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


if __name__ == "__main__":
    main()
