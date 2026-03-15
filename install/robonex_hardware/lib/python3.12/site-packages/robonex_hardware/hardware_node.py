import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
import json

class HardwareNode(Node):
    def __init__(self):
        super().__init__('hardware_node')
        
        self.port = '/dev/ttyUSB0'
        self.baudrate = 115200
        self.connected = False
        try:
            # import serial
            # self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            self.get_logger().info(f"Hardware connected on {self.port} (Mock Mode)")
        except Exception as e:
            self.get_logger().error(f" Hardware did not connect. Error: {e}")

        # 📡 Subscribe to the new topic
        self.subscription = self.create_subscription(
            String, 
            'robot_commands', 
            self.movement_callback,
            10
        )
        
        self.get_logger().info(" Hardware Node Started! Listening for JSON commands...")

    def movement_callback(self, msg):
        # 1.CHANGED: Unpack the JSON string back into a Python dictionary
        command = json.loads(msg.data)
        
        x = command.get('x', 0.0)
        y = command.get('y', 0.0)
        yaw = command.get('yaw', 0.0)
        stance = command.get('stance', 'stand')
        gait = command.get('gait', 'walk')
        speed_setting = command.get('speed', 'slow')
        
        # 2. CHANGED: Apply your dynamic Speed Math!
        if speed_setting == "fast":
            multiplier = 255
        elif speed_setting == "medium":
            multiplier = 127
        else:
            multiplier = 64
        
        speed_x = int(abs(x) * multiplier)
        speed_y = int(abs(y) * multiplier)
        speed_yaw = int(abs(yaw) * multiplier)

        dir_x = 1 if x >= 0 else -1
        dir_y = 1 if y >= 0 else -1
        dir_yaw = 1 if yaw >= 0 else -1

        # 3. Format the strict payload for the physical robot
        payload = f"CMD,{dir_x * abs(speed_x)},{dir_y * abs(speed_y)},{dir_yaw * abs(speed_yaw)},{stance},{gait}\n"
        
        # self.serial_conn.write(payload.encode()) # Uncomment to send to real hardware
        self.get_logger().info(f"Sent to Motors -> {payload.strip()}")

def main(args=None):
    rclpy.init(args=args)
    node = HardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down hardware node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()