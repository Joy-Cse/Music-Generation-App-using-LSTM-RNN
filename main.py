from sample import sample
import argparse
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST', 'GET'])
def generate():
    parser = argparse.ArgumentParser(description='Sample some text from the')
    parser.add_argument('epoch', type=int, help='epoch checkpoint to sample')
    parser.add_argument('--seed', default='', help='initial seed for the text')
    parser.add_argument('--len', type=int, default=512, help='no of character')
    args = parser.parse_args()
    print(args.epoch)
    music = sample(args.epoch, args.seed, args.len)
    # return sample(args.epoch, args.seed, args.len)
    return render_template('generate.html', data=music)


if __name__ == '__main__':
    app.run(debug=True)