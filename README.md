<div style="font-family: 'Times New Roman', serif;">

<h1>Nexus Clean: AI-Based Drug Trafficking Detection System</h1>

<h2>🧠 About the Project</h2>
<p>
<b>Nexus Clean</b> is an AI-driven intelligence system designed to detect, monitor, and analyze illegal 
drug trafficking and money laundering activities across encrypted and social media platforms. 
By leveraging Machine Learning, Natural Language Processing (NLP), and OSINT techniques, the 
system identifies suspicious messages, channels, and user handles on <b>Telegram</b> and <b>Instagram</b>.
</p>

<p>
This project is developed as a Final Year Engineering Project by:
<br><b>Kavya Parekh</b><br>
<b>Arpit Patel</b><br>
<b>Kavya Mehta</b><br>
</p>

<h2>⚙️ Key Features</h2>

<ul>
<li><b>Real-time scanning</b> of Telegram channels, chats, and Instagram DMs</li>
<li><b>AI-powered classification</b> using fine-tuned XLM-RoBERTa model</li>
<li><b>Automatic detection</b> of drug trafficking and money laundering patterns</li>
<li><b>Dynamic analytics dashboard</b> for live scans and insights</li>
<li><b>MongoDB-backed storage</b> for high-volume data handling</li>
<li><b>Scalable architecture</b> suitable for law enforcement integration</li>
</ul>

<h2>🏗 System Architecture</h2>

<pre>
Telegram/Instagram Data
        │
        ▼
Scraper Layer (Telethon / Session APIs)
        │
        ▼
Preprocessing Engine (spaCy, regex, timestamp normalization)
        │
        ▼
XLM-RoBERTa Classification (Drug / Money Laundering)
        │
        ▼
MongoDB Storage
        │
        ▼
Dashboard & Live Scan UI
</pre>

<h2>🛠 Technologies Used</h2>

<ul>
<li><b>AI & NLP:</b> PyTorch, XLM-RoBERTa, spaCy</li>
<li><b>OSINT Scraping:</b> Telethon, Instaloader, Requests</li>
<li><b>Backend:</b> Flask, JWT Authentication, APScheduler</li>
<li><b>Database:</b> MongoDB, BSON</li>
<li><b>Frontend:</b> HTML, CSS, JavaScript</li>
</ul>

<h2>🧪 How It Works</h2>

<ol>
<li>Scrapers fetch data from Telegram & Instagram</li>
<li>Preprocessing engine cleans and normalizes messages</li>
<li>ML model classifies messages into risk categories</li>
<li>MongoDB stores structured intelligence records</li>
<li>Dashboard visualizes insights in real time</li>
</ol>

<h2>📦 Setup Instructions</h2>

<p>Install requirements:</p>
<pre>pip install -r requirements.txt</pre>

<p>Start MongoDB:</p>
<pre>mongod</pre>

<p>Download spaCy model:</p>
<pre>python -m spacy download en_core_web_sm</pre>

<p>Run backend:</p>
<pre>python app.py</pre>

<p>Access dashboard:</p>
<pre>http://127.0.0.1:5000</pre>

<h2>🙏 Acknowledgements</h2>
<p>
Special thanks to our guide <b>Priti Bokariya Ma’am</b> for her continuous support, guidance, and motivation.<br>
Gratitude to our HOD <b>Janardan Kulkarni Sir</b> for his encouragement, vision, and belief in our project.
</p>

<h2>👥 Team</h2>
<ul>
<li><b>Kavya Parekh</b></li>
<li><b>Arpit Patel</b></li>
<li><b>Kavya Mehta</b></li>
</ul>

</div>
