# app.py
from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import bigquery
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import io
import base64
from urllib.parse import quote
from google.oauth2 import service_account
import os
import json

# Import your analyzer
from arxiv_analyzer import ArxivAnalyzer

app = Flask(__name__)

# Initialize BigQuery client and analyzer
def get_bigquery_client():
    """Initialize BigQuery client with service account credentials"""
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    
    # For production (Render)
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
        credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = bigquery.Client(project=project_id, credentials=credentials)
    # For local development
    elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        client = bigquery.Client(project=project_id)
    else:
        # Fallback - might not work without proper auth
        client = bigquery.Client(project=project_id)
    
    return client

client = get_bigquery_client()
analyzer = ArxivAnalyzer(client)

# Available domains
# AVAILABLE_DOMAINS = [
#     'hep-ex', 'hep-ph', 'hep-th', 'astro-ph', 'cond-mat', 'gr-qc',
#     'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics', 'quant-ph',
#     'cs', 'math', 'stat', 'eess', 'econ', 'q-bio', 'q-fin'
# ]

AVAILABLE_DOMAINS = ['hep-ex','cs','cs.AI']
# Plot types configuration
PLOT_TYPES = {
    'papers_per_month': {
        'name': 'Papers Per Month',
        'description': 'Number of papers published per month over time'
    },
    'avg_title_length': {
        'name': 'Average Title Length',
        'description': 'Average number of words in paper titles over time'
    },
    'avg_summary_length': {
        'name': 'Average Abstract Length', 
        'description': 'Average number of words in paper abstracts over time'
    },
    'avg_authors': {
        'name': 'Average Number of Authors',
        'description': 'Average number of authors per paper over time'
    },
    'top_words_title': {
        'name': 'Top Words in Titles',
        'description': 'Most frequently used words in paper titles'
    },
    'top_words_summary': {
        'name': 'Top Words in Abstract',
        'description': 'Most frequently used words in paper abstracts'
    },
    'top_bigrams_title': {
        'name': 'Top Bigrams in Titles',
        'description': 'Most frequently used word pairs in paper titles'
    },
    'top_bigrams_summary': {
        'name': 'Top Bigrams in Abstract', 
        'description': 'Most frequently used word pairs in paper abstracts'
    },
    'word_trends': {
        'name': 'Word Trends Over Time',
        'description': 'Track specific words frequency over time'
    },
    'bigram_trends': {
        'name': 'Bigram Trends Over Time',
        'description': 'Track specific bigrams frequency over time'
    }
}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Important: close figure to free memory
    return plot_url

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def index():
    """Main page with domain and plot selection"""
    return render_template('index.html', 
                         domains=AVAILABLE_DOMAINS, 
                         plot_types=PLOT_TYPES)

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    """Generate and return plot based on user selection"""
    try:
        data = request.get_json()
        domain = data.get('domain')
        plot_type = data.get('plot_type')
        
        # Validate inputs
        if domain not in AVAILABLE_DOMAINS:
            return jsonify({'error': 'Invalid domain selected'}), 400
        
        if plot_type not in PLOT_TYPES:
            return jsonify({'error': 'Invalid plot type selected'}), 400
        
        # Generate plot based on type
        fig = None
        plot_data = None
        
        if plot_type == 'papers_per_month':
            df = analyzer.create_papers_per_month_query(domain=domain)
            fig = analyzer.plot_per_month(df, 
                                        f'Papers Published Per Month in {domain.upper()}',
                                        'Number of Papers')
            
        elif plot_type == 'avg_title_length':
            df = analyzer.create_average_query(domain=domain, column='title')
            fig = analyzer.plot_per_month(df,
                                        f'Average Title Length in {domain.upper()}',
                                        'Average Words')
            
        elif plot_type == 'avg_summary_length':
            df = analyzer.create_average_query(domain=domain, column='summary')
            fig = analyzer.plot_per_month(df,
                                        f'Average Summary Length in {domain.upper()}',
                                        'Average Words')
            
        elif plot_type == 'avg_authors':
            df = analyzer.create_average_author_query(domain=domain)
            fig = analyzer.plot_per_month(df,
                                        f'Average Number of Authors in {domain.upper()}',
                                        'Average Authors')
            
        elif plot_type == 'top_words_title':
            df = analyzer.create_top_words_query(domain=domain, column='cleaned_title')
            fig = analyzer.plot_word_frequencies(df, 'cleaned_title', bigram=False)
            
        elif plot_type == 'top_words_summary':
            df = analyzer.create_top_words_query(domain=domain, column='cleaned_summary')
            fig = analyzer.plot_word_frequencies(df, 'cleaned_summary', bigram=False)
            
        elif plot_type == 'top_bigrams_title':
            df = analyzer.create_top_bigrams_query(domain=domain, column='cleaned_title')
            fig = analyzer.plot_word_frequencies(df, 'cleaned_title', bigram=True)
            
        elif plot_type == 'top_bigrams_summary':
            df = analyzer.create_top_bigrams_query(domain=domain, column='cleaned_summary')
            fig = analyzer.plot_word_frequencies(df, 'cleaned_summary', bigram=True)
            
        elif plot_type == 'word_trends':
            words = data.get('words', ['collision', 'decay', 'search'])
            column = data.get('column', 'cleaned_title')
            df = analyzer.create_specific_words_trend_query(domain=domain, 
                                                          column=column, 
                                                          words_to_plot=words)
            fig = analyzer.plot_specific_word_frequency_from_query(df, words, 
                                                                 f'Word Trends in {domain.upper()}',
                                                                 show_plot=False)
            
        elif plot_type == 'bigram_trends':
            bigrams = data.get('bigrams', ['higgs boson', 'machine learning', 'dark matter'])
            column = data.get('column', 'cleaned_summary')
            df = analyzer.create_specific_bigrams_trend_query(domain=domain,
                                                            column=column,
                                                            bigrams_to_plot=bigrams)
            fig = analyzer.plot_specific_word_frequency_from_query(df, bigrams,
                                                                 f'Bigram Trends in {domain.upper()}',
                                                                 show_plot=False,
                                                                 bigram=True)
        
        if fig is None:
            return jsonify({'error': 'Failed to generate plot'}), 500
        
        # Convert figure to base64
        plot_url = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'plot_type': PLOT_TYPES[plot_type]['name'],
            'domain': domain.upper()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating plot: {str(e)}'}), 500

@app.route('/api/domains')
def get_domains():
    """API endpoint to get available domains"""
    return jsonify(AVAILABLE_DOMAINS)

@app.route('/api/plot_types')
def get_plot_types():
    """API endpoint to get available plot types"""
    return jsonify(PLOT_TYPES)

if __name__ == '__main__':
    # for local testing
    #app.run(debug=True, host='0.0.0.0', port=5000) 
    # for production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)