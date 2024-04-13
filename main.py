from moviepy.editor import VideoFileClip

# uploading a video file
video = VideoFileClip("video_1.mp4")

# separating audio from video
audio = video.audio

'''
# saving audio to a separate file
audio.write_audiofile("audio.mp3")
'''

# let's define the loss function:
# insertions - inserting words that are not in the source text
# substitutions - incorrect word substitutions
# deletions - the system did not recognize the word / missed the word
def loss_func(insertions, substitutions, deletions, total_word):
    return (insertions + substitutions + deletions) / total_word

