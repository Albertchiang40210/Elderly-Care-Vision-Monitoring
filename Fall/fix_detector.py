with open("modules/fall_detector.py", "r") as f:
    content = f.read()

import re
# Find the dictionary
pattern = r'"kpts_sequence": collections.deque\(maxlen=30\),\s+"last_stable_action": "Tracking",\s+"prev_center": None\s+}'
replacement = r'"kpts_sequence": collections.deque(maxlen=30),\n                    "action_vote_buffer": collections.deque(maxlen=10),\n                    "last_stable_action": "Tracking",\n                    "prev_center": None\n                }'

content = re.sub(pattern, replacement, content)

with open("modules/fall_detector.py", "w") as f:
    f.write(content)
