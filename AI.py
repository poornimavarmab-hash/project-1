import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama



st.set_page_config(
    page_title="SmartTech Assist",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827, #172554);
        color: white;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.25),
            rgba(14, 165, 233, 0.15)
        );
        border: 1px solid rgba(96, 165, 250, 0.25);
        margin-bottom: 25px;
    }

    .main-header h1 {
        font-size: 42px;
        margin-bottom: 8px;
        color: #60a5fa;
    }

    .main-header p {
        font-size: 18px;
        color: #cbd5e1;
    }

    /* Cards */
    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 20px;
    }

    .info-card h3 {
        color: #60a5fa;
    }

    /* Solution box */
    .solution-box {
        padding: 25px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(96, 165, 250, 0.35);
        margin-top: 20px;
    }

    /* Status */
    .status-box {
        padding: 12px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: #94a3b8;
        font-size: 14px;
    }

    /* Text area */
    textarea {
        border-radius: 15px !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)




st.markdown("""
<div class="main-header">

  

    
        AI-Powered Computer Troubleshooting & Support System


</div>
""", unsafe_allow_html=True)




with st.sidebar:

    st.markdown("##  SmartTech Assist ⚙️")

    st.markdown("𓂃🪶")

    st.markdown("### Technologies 🤖 ")

    st.write("🐍 Python")
    st.write("🎨 Streamlit")
    st.write("🧠 Sentence Transformer")
    st.write("🗄️ ChromaDB")
    st.write(" Ollama + Llama 3.2 ⛓️‍💥")

    st.markdown("---")

    st.markdown("### 💡 Supported Problems")

    st.write("📶 WiFi & Network")
    st.write("🐌 Slow Computer")
    st.write("🔋 Battery Problems")
    st.write("🌡️ Overheating")
    st.write("⌨️ Keyboard")
    st.write("🖱️ Mouse")
    st.write("🔊 Audio")
    st.write("⚡ Power Problems")
    st.write(" Blue Screen")
    st.write("🖨️ Printer")

    st.markdown("---")

    st.markdown("""
    <div class="status-box">
    🟢 SmartTech Assist Ready
    </div>
    """, unsafe_allow_html=True)




st.markdown("""
<div class="info-card">

<h3>🔧 How can I help?</h3>

<p>
Describe your computer or laptop problem in simple words.
SmartTech Assist will analyze your problem, search its
troubleshooting knowledge base and generate a step-by-step
solution using AI.
</p>

</div>
""", unsafe_allow_html=True)




@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("🧠 Loading AI model..."):
    embedding_model = load_embedding_model()




@st.cache_resource
def create_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="troubleshooting"
    )

    problems = [

        {
            "id": "1",
            "problem": "WiFi is not connecting",
            "solution": """
1. Check whether WiFi is turned on.
2. Turn WiFi off and turn it on again.
3. Restart the router.
4. Forget the WiFi network and reconnect.
5. Restart the computer.
"""
        },

        {
            "id": "2",
            "problem": "Computer is very slow",
            "solution": """
1. Restart the computer.
2. Close unnecessary applications.
3. Check Task Manager for high CPU or RAM usage.
4. Delete unnecessary temporary files.
5. Disable unnecessary startup applications.
6. Scan the computer for malware.
"""
        },

        {
            "id": "3",
            "problem": "Laptop battery is draining quickly",
            "solution": """
1. Reduce screen brightness.
2. Close unnecessary background applications.
3. Turn on battery saver.
4. Check applications consuming battery.
5. Update the operating system and drivers.
"""
        },

        {
            "id": "4",
            "problem": "Computer is overheating",
            "solution": """
1. Make sure air vents are not blocked.
2. Keep the laptop on a hard flat surface.
3. Clean dust from cooling vents.
4. Close applications using high CPU.
5. Check whether the cooling fan is working.
"""
        },

        {
            "id": "5",
            "problem": "Keyboard is not working",
            "solution": """
1. Restart the computer.
2. Check the keyboard connection.
3. Try another USB port.
4. Check the keyboard driver.
5. Test another keyboard.
"""
        },

        {
            "id": "6",
            "problem": "Mouse is not working",
            "solution": """
1. Check the mouse connection.
2. Try another USB port.
3. Replace the battery if wireless.
4. Restart the computer.
5. Update the mouse driver.
"""
        },

        {
            "id": "7",
            "problem": "No sound from computer",
            "solution": """
1. Check the volume level.
2. Make sure the computer is not muted.
3. Check the selected audio output device.
4. Restart the computer.
5. Update the audio driver.
"""
        },

        {
            "id": "8",
            "problem": "Computer does not turn on",
            "solution": """
1. Check whether the charger is connected.
2. Check the power socket.
3. Try another charger if available.
4. Disconnect external devices.
5. Hold the power button for several seconds.
6. Contact a technician if the problem continues.
"""
        },

        {
            "id": "9",
            "problem": "Blue screen error",
            "solution": """
1. Restart the computer.
2. Note the error code.
3. Update Windows and device drivers.
4. Remove recently installed problematic software.
5. Run Windows system checks.
6. Contact technical support if the problem continues.
"""
        },

        {
            "id": "10",
            "problem": "Printer is not printing",
            "solution": """
1. Check printer power.
2. Check USB or WiFi connection.
3. Select the correct printer.
4. Check paper and ink.
5. Restart the printer.
6. Update the printer driver.
"""
        }
    ]

    # Add documents only once

    if collection.count() == 0:

        documents = [
            item["problem"] + "\n" + item["solution"]
            for item in problems
        ]

        ids = [
            item["id"]
            for item in problems
        ]

        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

    return collection


collection = create_database()




st.markdown("### 📝 Describe Your Computer Problem")

user_problem = st.text_area(
    "Tell us what is happening...",
    placeholder=(
        "Example: My laptop is connected to WiFi "
        "but the internet is not working."
    ),
    height=150
)




category = st.selectbox(
    "📂 Select Problem Category",
    [
        "Auto Detect",
        "Network / WiFi",
        "Performance",
        "Battery",
        "Overheating",
        "Keyboard / Mouse",
        "Audio",
        "Power",
        "System Error",
        "Printer"
    ]
)




col1, col2 = st.columns(2)

with col1:

    troubleshoot = st.button(
        "🔍 Troubleshoot Problem",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🧹 Clear",
        use_container_width=True
    )


if clear:

    st.rerun()




if troubleshoot:

    if user_problem.strip() == "":

        st.warning(
            "⚠️ Please describe your computer problem first."
        )

    else:

        with st.spinner(
            "🔎 Analyzing your problem..."
        ):

            # Create embedding

            query_embedding = embedding_model.encode(
                user_problem
            ).tolist()

            # Search database

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            relevant_documents = results["documents"][0]

            # Combine retrieved knowledge

            context = "\n\n".join(
                relevant_documents
            )

       

        prompt = f"""
You are SmartTech Assist, an AI computer troubleshooting assistant.

User Problem:
{user_problem}

Selected Category:
{category}

Relevant troubleshooting knowledge:
{context}

Give a clear and beginner-friendly answer.

Use this format:

### 🔍 Possible Cause
Explain the likely cause.

### 🛠️ Recommended Solution
Give 4 to 6 simple steps.

### ⚠️ Warning
Explain when the user should contact a technician.

Only use information supported by the provided knowledge.
Do not invent hardware-specific information.
"""

       
        try:

            with st.spinner(
                "🤖 Generating AI solution..."
            ):

                response = ollama.chat(

                    model="llama3.2",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

            answer = response["message"]["content"]


            st.markdown(
                '<div class="solution-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "## 🛠️ Troubleshooting Result"
            )

            st.markdown(answer)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            

            with st.expander(
                "📚 View Retrieved Knowledge"
            ):

                for i, document in enumerate(
                    relevant_documents,
                    start=1
                ):

                    st.markdown(
                        f"### 🔹 Result {i}"
                    )

                    st.write(document)

        except Exception as e:

            st.error(
                "❌ Unable to connect to Ollama."
            )

            st.info(
                "Make sure Ollama is running and "
                "Llama 3.2 is installed."
            )

            st.code(
                "ollama pull llama3.2"
            )




st.markdown(
    """
    <div class="footer">

    💻 <b>SmartTech Assist</b><br>

    AI-Powered Computer Troubleshooting System<br>

    Built using Python • Streamlit • Sentence Transformers •
    ChromaDB • Ollama

    </div>
    """,
    unsafe_allow_html=True
)