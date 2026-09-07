# JIANMAN POLICY |  Absolute EEF control
python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 1920, "height": 1080, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Put all blocks into the box."   --actions_per_chunk=30   --duration_s=300 --record_dataset=true

python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 1920, "height": 1080, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Clean the table using the dust pan."   --actions_per_chunk=30   --duration_s=300 --record_dataset=true

python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 1920, "height": 1080, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Transfer the egg from the pan into the bowl."   --actions_per_chunk=30   --duration_s=300 --record_dataset=true

# D405 Camera OOD: 260322276290

python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "260322276290", "width": 1280, "height": 720, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Put all blocks into the box."   --actions_per_chunk=30   --duration_s=300 --record_dataset=true

python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "260322276290", "width": 1280, "height": 720, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Clean the table using the dust pan."   --actions_per_chunk=30   --duration_s=300 --record_dataset=true

python scripts/eval_yam_http_policy.py   --robot.left_arm_port=1235 --robot.right_arm_port=1234   --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "260322276290", "width": 1280, "height": 720, "fps": 30}
}'   --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act   --task="Transfer the egg from the pan into the bowl."  --actions_per_chunk=30   --duration_s=300 --record_dataset=true