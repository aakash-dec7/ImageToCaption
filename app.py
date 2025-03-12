import os
from flask import Flask, render_template, request, url_for
from src.imgtocap.prediction.prediction import Prediction
from src.imgtocap.config.configuration import ConfigurationManager


app = Flask(__name__)

# Load Prediction Model ONCE
prediction_model = Prediction(config=ConfigurationManager().get_prediction_config())


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption():
    try:
        if "image" not in request.files:
            return render_template("index.html", caption_text="No image uploaded.")

        image = request.files["image"]

        if image.filename == "":
            return render_template("index.html", caption_text="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

        # Generate caption
        caption_text = prediction_model.predict(image_path)

        # Convert file path to URL
        image_url = url_for("static", filename=f"uploads/{image.filename}")

        return render_template(
            "index.html", caption_text=caption_text, image_url=image_url
        )

    except Exception as e:
        return render_template("index.html", caption_text=f"Error: {str(e)}")


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=3000, debug=True)
    app.run(host="0.0.0.0", port=8080)
