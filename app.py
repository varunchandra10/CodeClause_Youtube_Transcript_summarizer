# app.py
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
api = Api(app)

class Transcript(Resource):
    def get(self):
        youtube_url = request.args.get('youtube_url')  # Get the youtube_url from the query parameters
        if not youtube_url:
            return {'message': 'Please provide a valid YouTube URL using the "youtube_url" query parameter.'}, 400

        try:
            # Extract YouTube video ID from the URL
            video_id = self.extract_video_id(youtube_url)
            
            # Fetch the transcript from the YouTube video ID
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript])

            # Perform text summarization using T5 model
            summarized_transcript = self.summarize_text(transcript_text)
            
            return {'transcript': summarized_transcript}, 200
        except Exception as e:
            return {'message': 'Error fetching or summarizing transcript.', 'error': str(e)}, 500

    def extract_video_id(self, youtube_url):
        # Extract the video ID from the YouTube URL
        # This is a simple example, you might want to enhance this function to handle different URL formats
        video_id = youtube_url.split('v=')[-1]
        return video_id

    def summarize_text(self, text):
        # Instantiate T5 tokenizer and model
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Define the transcript that should be summarized
        input_text = "summarize: " + text

        # Encode the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)

        # Decode the summary and return it as a string
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summarized_text

api.add_resource(Transcript, '/api/summarize')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
