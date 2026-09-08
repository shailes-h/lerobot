lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras='{
top: {"type": "intelrealsense", "serial_number_or_name": "262522075787", "width": 1280, "height": 720, "fps": 15, "use_depth": True, "align_depth_to_color": True}
}' \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port 5002 \
  --teleop.right_arm_port 5001 \
  --dataset.repo_id=local/PnP_v2 \
  --dataset.root=./datasets/PnP_v2 \
  --dataset.fps=15 \
  --dataset.num_episodes=30 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=30 \
  --dataset.video_encoding_batch_size=30 \
  --dataset.single_task="Put apple into the pan and the can into the bowl." \
  --dataset.push_to_hub=false \
  --display_data=false  \
  --robot.record_eef_pose=true



# EVAL
python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.cameras='{top: {"type": "intelrealsense", "serial_number_or_name": "262522075787", "width": 1280, "height": 720, "fps": 15, "use_depth": true, "align_depth_to_color": true}}' \
  --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act \
  --task="Put apple into the pan and the can into the bowl." \
  --actions_per_chunk=16 \
  --duration_s=300
