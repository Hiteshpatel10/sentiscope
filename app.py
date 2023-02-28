from flask import Flask, render_template, request, redirect, url_for
from flask_paginate import Pagination, get_page_args
from youtube_main import youtubeMain
from amazon_main import amazonMain
from utility.youtube.youtube_comments import extract_video_id


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
    link = request.form['link']
    videoId = extract_video_id(link)
    youtubeMain(videoId)
    return redirect(url_for("youtubeSentiResult", videoId=videoId))
  else:
    return render_template("youtube.html")
  
@app.route("/youtubeSentiResult")
def youtubeSentiResult():
  return render_template("youtube_result.html")


  
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

