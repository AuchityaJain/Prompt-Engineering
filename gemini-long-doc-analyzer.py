import os
from dotenv import load_dotenv
import tiktoken 
import time 

# CRITICAL FIX: Use the new, recommended Google GenAI SDK.
from google import genai
from google.genai import types

# --- Configuration & Initialization ---
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# --- Discover and Select Available Model (Logic updated for new SDK) ---
print("\n--- Discovering available Gemini models ---")
available_models = []
try:
    for m in client.models.list():
        available_models.append(m.name)
        
except Exception as e:
    print(f"Error listing models: {e}")
    print("Please check your API key, network connection, and Google Cloud project setup.")
    exit()

priority_models = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-pro',
]

LLM_MODEL = None
for p_model_name in priority_models:
    if p_model_name in available_models or f'models/{p_model_name}' in available_models:
        LLM_MODEL = p_model_name
        break

if not LLM_MODEL and available_models:
    LLM_MODEL = next((m.replace('models/', '') for m in available_models if 'gemini' in m and 'generateContent' in m), None)
    if LLM_MODEL:
        print(f"\nNo preferred model found. Selected first available Gemini model: {LLM_MODEL}")
    else:
        print("No suitable model could be selected. Exiting.")
        exit()

if LLM_MODEL:
    print(f"\nSelected model for content generation: {LLM_MODEL}")
else:
    print("No suitable model could be selected. Exiting.")
    exit()

# --- Configuration ---
MAX_TOKENS_PER_CHUNK = 10000
OVERLAP_TOKENS = 500
MAX_TOKENS_FOR_SUMMARY = 4096
MAX_TOKENS_FOR_FINAL_ANSWER = 1024

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Estimates token count using an OpenAI tokenizer as a proxy."""
    return len(encoding.encode(text))

# --- Core LLM Interaction Functions ---

def get_completion(prompt, model=LLM_MODEL, temperature=0.0, max_tokens_out=MAX_TOKENS_FOR_FINAL_ANSWER):
    """Helper function to get completion from Gemini API (using the new SDK)"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt, 
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens_out
            )
        )
        
        if response.text:
            return response.text
        else:
            feedback = response.prompt_feedback
            if feedback and feedback.block_reason != types.FinishReason.SAFETY:
                if response.candidates and response.candidates[0].finish_reason != types.FinishReason.STOP:
                    print(f"  Gemini blocked response. Reason: {response.candidates[0].finish_reason}")
                    return "ERROR: Gemini blocked or stopped generation prematurely."
            
            return "ERROR: Gemini generated an empty response (text field was empty)."

    except Exception as e:
        print(f"Error calling Gemini LLM: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            print("  This error is likely due to exceeding your API quota.")
        return "ERROR: Could not get LLM response."

print("LLM interaction functions defined.")
print("---")

def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap_tokens=OVERLAP_TOKENS):
    """
    Splits a long text into chunks based on token limits with overlap.
    Aims to split at natural sentence breaks if possible.
    """
    words = text.split()
    chunks = []
    current_chunk_words = []
    current_chunk_tokens = 0

    avg_tokens_per_word = count_tokens(text) / len(words) if words else 1 
    
    for word in words:
        word_tokens_estimate = max(1, int(avg_tokens_per_word)) 
        
        if current_chunk_tokens + word_tokens_estimate > max_tokens:
            chunks.append(" ".join(current_chunk_words))
            
            overlap_word_count = int(overlap_tokens / avg_tokens_per_word) 
            overlap_start_index = max(0, len(current_chunk_words) - overlap_word_count)
            
            current_chunk_words = current_chunk_words[overlap_start_index:]
            current_chunk_tokens = count_tokens(" ".join(current_chunk_words))
            
            current_chunk_words.append(word)
            current_chunk_tokens += word_tokens_estimate
        else:
            current_chunk_words.append(word)
            current_chunk_tokens += word_tokens_estimate

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

print("Document chunking function defined.")
print("---")

def summarize_chunks(chunks):
    """Summarizes each text chunk using an LLM."""
    summaries = []
    print(f"\nSummarizing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)} (Estimated Tokens: {count_tokens(chunk)})...")
        summary_prompt = f"""
        Please provide a concise summary of the following text. Focus on the main ideas and key information.
        The resulting summary MUST capture all critical details needed to answer complex questions later.
        
        TEXT:
        {chunk}
        
        COMPREHENSIVE SUMMARY:
        """
        summary = get_completion(summary_prompt, max_tokens_out=MAX_TOKENS_FOR_SUMMARY) 
        summaries.append(summary)
        
        if summary.startswith("ERROR:"):
             print(f"  Received error while summarizing chunk {i+1}. Stopping summarization.")
             return summaries
             
        time.sleep(0.1)
    print("Chunk summarization complete.")
    return summaries

print("Chunk summarization function defined.")
print("---")

def query_combined_summary(combined_summary, question):
    """Queries the combined summary with a specific question."""
    print(f"\nQuerying the combined summary (Estimated Tokens: {count_tokens(combined_summary)})...")
    final_prompt = f"""
    Based on the following summarized information, answer the question below.
    If the information is not present in the summary, state that explicitly.

    SUMMARIZED INFORMATION:
    {combined_summary}

    QUESTION:
    {question}

    ANSWER:
    """
    answer = get_completion(final_prompt, max_tokens_out=MAX_TOKENS_FOR_FINAL_ANSWER)
    print("Final query complete.")
    return answer

print("Final querying function defined.")
print("---")


if __name__ == "__main__":
    # --- RESTORED EXAMPLE LONG DOCUMENT ---
    long_document_content = """
    Artificial intelligence (AI) ethics is a rapidly evolving field grappling with the moral implications of intelligent autonomous systems. As AI permeates various aspects of society, from healthcare diagnostics to financial trading and autonomous vehicles, the ethical dilemmas surrounding its development and deployment become increasingly critical. Key concerns include algorithmic bias, privacy violations, accountability for AI actions, and the impact on employment.

    Algorithmic bias arises when AI systems perpetuate or even amplify existing societal prejudices. This can occur through biased training data, where historical human decisions, often flawed, are learned by the AI. For instance, facial recognition systems have shown higher error rates for certain demographic groups, leading to unjust outcomes. Similarly, AI used in hiring processes can inadvertently favor certain applicants based on patterns in past successful hires, which might reflect historical biases rather than true merit. Addressing algorithmic bias requires careful data curation, fairness-aware machine learning techniques, and diverse development teams.

    Privacy is another paramount concern. AI systems often rely on vast amounts of personal data for training and operation. The collection, storage, and processing of this data raise questions about informed consent, data anonymization, and the potential for re-identification. Regulations like GDPR and CCPA aim to provide frameworks for data protection, but the rapid advancements in AI continually challenge these legal boundaries. Differential privacy and federated learning are emerging techniques to enable AI development while preserving individual privacy.

    Accountability in AI systems is complex. When an autonomous vehicle causes an accident, or an AI-driven medical device makes an incorrect diagnosis, who is responsible? Is it the developer, the deployer, the user, or the AI itself? Traditional legal frameworks are often ill-equipped to handle the distributed agency of AI systems. Establishing clear lines of accountability is vital for public trust and for fostering responsible innovation.

    The impact of AI on employment is a widely debated topic. While AI is expected to automate many routine tasks, potentially displacing workers, it also promises to create new jobs and enhance human productivity. The ethical challenge lies in ensuring a just transition for affected workers, providing opportunities for reskilling and upskilling, and addressing potential widening economic inequalities. Universal Basic Income (UBI) is often discussed as a policy response to potential widespread automation.

    Furthermore, the potential for malicious use of AI, such as autonomous weapons or sophisticated disinformation campaigns, poses existential threats. The development of ethical guidelines for AI in warfare (e.g., the principle of human control over lethal autonomous weapons) is an active area of international discussion.

    The future of AI ethics involves moving beyond reactive problem-solving to proactive design-for-ethics. This includes embedding ethical principles into AI development from the outset, encouraging interdisciplinary collaboration between AI researchers, ethicists, sociologists, and policymakers, and promoting public education and engagement. The goal is not to halt AI progress but to steer it towards a future that benefits all of humanity, upholding values of fairness, transparency, and human flourishing.
    
    One challenge in AI ethics is the "black box" problem, where complex deep learning models make decisions in ways that are opaque to human understanding. This lack of interpretability can hinder efforts to diagnose and correct biases or to establish accountability. Researchers are actively working on Explainable AI (XAI) techniques to provide insights into how AI models arrive at their conclusions. These techniques include saliency maps, LIME (Local Interpretable Model-agnostic Explanations), and SHAP (SHapley Additive exPlanations), which aim to make AI decisions more transparent.

    Another emerging area is the ethics of superintelligence. While still largely theoretical, the hypothetical development of AI far surpassing human intelligence raises profound questions about control, alignment of goals, and the very future of humanity. Ensuring that such advanced AI systems are designed with human values and safety as core priorities is a long-term, complex challenge. Research into AI alignment seeks to solve this problem by ensuring that highly intelligent AI systems act in accordance with human intentions and well-being. This involves developing robust methods for value learning, trustworthy AI, and robust AI safety protocols.

    The growth of AI in creative fields also presents new ethical quandaries. Questions arise regarding intellectual property ownership when AI generates art, music, or literature, especially if trained on copyrighted materials. The authenticity and value of human creativity are also debated in the face of increasingly sophisticated AI-generated content. Furthermore, the potential for deepfakes and synthetic media to spread misinformation or manipulate public opinion is a significant societal risk that demands careful ethical consideration and technological countermeasures. These issues necessitate discussions on attribution, transparency in AI-generated content, and the development of ethical guidelines for creative AI applications.

    The development and deployment of AI in military applications is a particularly contentious area. The concept of Lethal Autonomous Weapons Systems (LAWS), often dubbed "killer robots," raises fundamental ethical questions about human control over life-and-death decisions, the potential for escalation in conflicts, and the blurring of moral responsibility. International efforts are underway to establish norms and regulations to govern the use of AI in warfare, advocating for meaningful human control over critical functions. The dual-use nature of many AI technologies—beneficial in civilian applications but potentially harmful in military contexts—further complicates the ethical landscape.

    Finally, the environmental impact of large-scale AI training and deployment is gaining attention. Training large language models and complex neural networks consumes significant amounts of energy, contributing to carbon emissions. The demand for rare earth minerals in AI hardware also raises concerns about resource extraction and waste management. Ethical considerations in this domain involve promoting energy-efficient AI architectures, developing sustainable hardware, and prioritizing AI research that minimizes its ecological footprint. Balancing the benefits of AI with its environmental costs is an emerging challenge for responsible AI development.
    """

    # We'll duplicate the content a few times to ensure it's sufficiently long
    long_document_content = long_document_content * 3

    question = "What are the main ethical challenges discussed regarding AI across all its applications, and what solutions or approaches are mentioned?"
    
    print(f"Original document estimated token count: {count_tokens(long_document_content)}")

    # 1. Chunk the Document
    chunks = chunk_text(long_document_content)
    print(f"Document split into {len(chunks)} chunks.")
    print("---")

    # 2. Summarize Each Chunk
    individual_summaries = summarize_chunks(chunks)

    # 3. Combine Summaries
    combined_summary_text = "\n\n".join(individual_summaries)
    
    if combined_summary_text.startswith("ERROR:"):
         print(f"FATAL ERROR: Summarization failed. Cannot proceed to final query.")
         print(combined_summary_text)
         exit()
         
    print(f"\nCombined summary estimated token count: {count_tokens(combined_summary_text)}")
    print("---")

    # 4. Query the Combined Summary
    final_answer = query_combined_summary(combined_summary_text, question)

    print("\n--- Final Answer from Gemini LLM (Comprehensive) ---")
    print(final_answer)

    print("\n--- Original Question ---")
    print(question)
    
    print("\n--- Snippet of Combined Summary (for context) ---")
    print(combined_summary_text[:2000] + "\n...")
