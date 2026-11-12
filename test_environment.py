from environment import WarehouseEnvironment

env = WarehouseEnvironment()

state = env.reset()
print("Starting state:", state)

# Move right
state, reward, done = env.step(3)

print("New state:", state)
print("Reward:", reward)
print("Done:", done)

# Attempt to pick up a package
state, reward, done = env.step(4)

print("Pickup reward:", reward)
