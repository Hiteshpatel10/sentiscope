from flask import Flask, render_template, request, redirect, url_for
from googleapiclient.discovery import build
import pandas as pd
from flask_paginate import Pagination, get_page_args

app = Flask(__name__,template_folder="templates")
app.config['SECRECT_KEY'] = "SentiScope"

@app.route("/")
def home():
  return render_template("home.html")

# @app.route("/youtube", methods=["POST","GET"])
# def hitesh():
#     df = pd.read_csv('data/new.csv')
#     page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
#     if per_page is None:
#       per_page = 20
#     if request.method == "POST":
#       videoUrl = request.form['link']
#       videoId = extract_video_id(videoUrl)
#       yt_service = build("youtube", "v3", developerKey="AIzaSyAOvxr2XvdsjlqFnc-E0EgKbzlEfXb88_4")
#       comments_dict = get_comments(yt_service, videoId)
#       save_to_csv(comments_dict, 'new')
#       df = pd.read_csv('data/new.csv')
#       data = df[offset: offset + per_page]
#       pagination = Pagination(page=page, per_page=per_page, total=len(df), css_framework='bootstrap4')
#       return render_template('csv_reader.html', data_var=data.to_html(), pagination=pagination)
#     else: 
#       data = df[offset: offset + per_page]
#       pagination = Pagination(page=page, per_page=per_page, total=len(df), css_framework='bootstrap4')
#       return render_template('csv_reader.html', data_var=data.to_html(), pagination=pagination)

@app.route("/youtube", methods=["POST","GET"])
def youtube():

  if request.method == "POST":
    videoUrl = "this"
    return redirect(url_for("youtubeResult", videoUrl=videoUrl))
  else:
    return render_template("youtube.html")


@app.route("/<videoUrl>")
def youtubeResult(videoUrl):
  return render_template("youtubeResult.html", videoUrl=videoUrl)


if __name__ == "__main__":
  app.run(debug=True)

