This Python project, which could be aptly titled Gemini Long Document Analyzer, is designed to efficiently process and query extremely long texts that exceed the context window limitations of large language models (LLMs).

It implements a MapReduce-style workflow for long-form content analysis:

Initialization: Uses the new google-genai SDK and environment variables (.env) to initialize the Gemini client and select the best available model (prioritizing gemini-2.5-flash).

Chunking: The input document is split into smaller, manageable pieces (chunks) with a configurable token limit and overlap to preserve context across boundaries.

Mapping (Summarization): Each individual chunk is passed to the Gemini LLM for a comprehensive summary (the "Map" phase).

Reducing (Final Query): All the individual summaries are concatenated into one combined summary. This combined text is then queried with the user's final question to generate a concise, accurate answer (the "Reduce" phase).

This structure ensures that complex questions can be answered accurately, even when the source material is far too long to fit into a single API call.
