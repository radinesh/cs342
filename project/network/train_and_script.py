from model import *


model = Imitation()

train(model)

scripted_model = torch.jit.script(model)

torch.jit.save(scripted_model, "state_agent.pt")