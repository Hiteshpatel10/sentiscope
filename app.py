from flask import Flask, render_template, request, redirect, url_for
from flask_paginate import Pagination, get_page_args
from youtube_main import youtubeMain
from amazon_main import amazonMain
from utility.youtube.youtube_comments import extract_video_id
import pandas as pd
import uuid


app = Flask(__name__,template_folder="templates")
app.config['SECRECT_KEY'] = "SentiScope"
APP_UUID = uuid.uuid4()


@app.route("/")
def home():
  return render_template("home.html")

import os

@app.route("/csv/<path:file_name>", methods=["GET"])
def show_csv(file_name):
    # Read the CSV file
    file_path = os.path.join( 'data', file_name)
    df = pd.read_csv(file_path)

    # Paginate the data
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    data = df[offset: offset + per_page]
    pagination = Pagination(page=page, per_page=per_page, total=len(df), css_framework='bootstrap4')

    # Render the HTML table
    return render_template('csv_reader.html', data_var=data.to_html(index=False), pagination=pagination)

@app.route("/youtube", methods=["POST","GET"])
def youtube():
  if request.method == "POST":
    link = request.form['link']
    videoId = extract_video_id(link)
    youtubeMain(videoId, APP_UUID)
    return redirect(url_for("youtubeSentiResult", videoId=videoId))
  else:
    return render_template("youtube.html")
  
@app.route("/youtubeSentiResult")
def youtubeSentiResult():
  return render_template("youtube_result.html", uuid = str(APP_UUID))


  
@app.route("/amazon", methods=["POST","GET"])
def amazon():
  if request.method == "POST":
    link = request.form['link']
    videoId = extract_video_id(link)
    amazonMain()
    return redirect(url_for("amazonSentiResult", videoId=videoId))
  else:
    return render_template("youtube.html")

@app.route("/amazonSentiResult")
def amazonSentiResult():
  return render_template("amazon_result.html")


if __name__ == "__main__":
  app.run(debug=True)

