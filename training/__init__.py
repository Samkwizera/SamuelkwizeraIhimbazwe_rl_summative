import os

# torchrl warns about every batch size that does not divide the frame budget exactly
os.environ.setdefault("RL_WARNINGS", "False")
