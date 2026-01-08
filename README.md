# Topic Analysis Dashboard

An interactive Streamlit dashboard for analyzing topics with filtering capabilities by date, level1, and level2 categories.

## Features

- 📊 **Interactive Dashboard**: Beautiful Streamlit web interface
- 🔍 **Advanced Filtering**: Filter by date range, level1, and level2 categories
- 📈 **Top 10 Topics Visualization**: Bar charts showing top topics by raw_conversations
- 📋 **Data Tables**: Detailed tables with metrics
- 💾 **Data Export**: Download filtered data as CSV

## Installation

1. Clone this repository:
```bash
git clone https://github.com/litancherikover/Topic_analysis.git
cd Topic_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard (Recommended)

Run the interactive Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Command Line Tool

Alternatively, use the command-line tool:
```bash
python topic_analyzer.py
```

## Data Format

The tool expects a CSV file with the following columns:
- `device`: Device type
- `country`: Country code
- `topic`: Topic name
- `level1`: Top-level category
- `level2`: Subcategory
- `raw_messages`: Number of messages
- `raw_users`: Number of users
- `raw_conversations`: Number of conversations (used for ranking)
- `date`: Date in YYYY-MM-DD format

## Project Structure

```
.
├── streamlit_app.py      # Main Streamlit dashboard
├── topic_analyzer.py     # Command-line analysis tool
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Requirements

- Python 3.9+
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- streamlit >= 1.28.0

## License

MIT License
