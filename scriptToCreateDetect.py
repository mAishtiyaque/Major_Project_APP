from tensorflow.keras.models import load_model
model = load_model('77model1743-0604.h5')
model_json = model.to_json()
with open("detect.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("detect.h5")
print("\n>>> Saved model to disk")