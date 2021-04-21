from flask import Flask, request, url_for, redirect, render_template
import os
from test import Tester
import torch
import wandb
from create_bar_dataset import NoteRepresentationManager

app = Flask(__name__)


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    # with torch.no_grad():
    #     tester.generate(nm)
    return render_template("results.html")


@app.route("/")
def index():
    return render_template("index.html")


checkpoint_name = os.path.join("remote", "fix")
tester = Tester(torch.load(checkpoint_name + os.sep + "encoder.pt"),
                torch.load(checkpoint_name + os.sep + "latent_compressor.pt"),
                torch.load(checkpoint_name + os.sep + "latent_decompressor.pt"),
                torch.load(checkpoint_name + os.sep + "decoder.pt"),
                torch.load(checkpoint_name + os.sep + "generator.pt"))
nm = NoteRepresentationManager()

if __name__ == "__main__":
    wandb.init()
    wandb.unwatch()
    app.run(debug=True)

