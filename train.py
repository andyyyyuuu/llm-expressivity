from nnsight import LanguageModel

model = LanguageModel("meta-llama/Llama-3.2-1B", device_map="auto", dispatch=True)

# toy intervention for now

with model.trace("Hello"):
    model.model.layers[5].output[0][:] = 0
    logits = model.output.save()

print(logits.logits.shape)
