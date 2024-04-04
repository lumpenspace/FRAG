# FRAG: Focused Retrieval-Augmented Generation

FRAG is a novel approach to retrieval-augmented generation that addresses key limitations of existing RAG (Retrieval Augmented Generation) systems. By decoupling the retrieval and generation phases, FRAG enables more flexible, efficient, and context-aware generation from large unstructured knowledge sources.

It is part of a process of rewriting and modularisation for [raft](https://github.com/lumpenspace/raft).

## Motivation

While RAG systems have shown promise in knowledge-intensive tasks, they suffer from several limitations:

1. **Overconfidence**: RAG models often overestimate the relevance of retrieved results, taking them for granted without critically assessing their applicability to the current context.
2. **Stylistic inconsistency**: Retrieved passages can have varying styles that differ from the main generation model, leading to inconsistent outputs. Rephrasing sources before storage can mitigate this but may result in information loss.
3. **Extraneous information**: Retrieved results, while relevant, may be overly long and contain extraneous details, polluting the context and hindering focused generation.
4. **Inefficiency**: Techniques like scratchpad generation are token-intensive and introduce response delays.

FRAG addresses these issues by introducing a focused retrieval phase that identifies concise, relevant knowledge fragments, followed by a generation phase that attends to these fragments to produce coherent, context-aware outputs.

## Approach

### Summary

Key components:

- Embeddings Database
- Interface Model
- Multiple Archivists

Flow:

1. Fetch closest N fragments meeting a threshold from embeddings db.
2. Perform focused retrieval via Archivists
3. Archivists summarize helpful fragments
4. Feed summaries to Interface Model for context-aware generation
5. Generate final response

<div class="mermaid">
   graph TD
      A[User Input] --> |Last question| E[context]
      B[Vector DB] --> |N closest fragments| C[Archivist 1]
      B --> |N closest fragments| D[Archivist 2]
      B --> |N closest fragments| F[Archivist N]
      
      C --> |Relevant summary| G[Merged summary]
      D --> |Relevant summary| G
      F --> |Relevant summary| G
      
      A --> E[Context]
      E --> |Previous interaction| B
      E --> H[Interface Model]
      G --> H
</div>

### Architecture

FRAG is composed by three main elements:

- An embeddings databases containing the RAG corpus, as-is.
- An Interface Model, which will interact directly with the user. This should be the most powerful and capable model
- One or more Archivist

For each interaction, the system will fetch the closest N fragments (over a proximity threshold P) to the last model and user outputs.

At this point, there will be two generation phases

1. **Focused Retrieval**:
   - N instances of the Archivist are instantiated. Each is given the previous interaction, current question, and is briefed with determining whether the retrieved document can help answer the question.
     - In case of negative response, that Archivist won't relay an answer.
     - Otherwise, it will provide a summary of the document targeted specifically for the Persona embodied by the Interface Model to answer the last question.
     - A pointer to the source document is also added to the response.

2. **Context-Aware Generation**:
   - Retrieved and summarized fragments are fed as additional context to the Interface Model
   - The model attends to these focused knowledge snippets to inform its generation process
   - Fragments serve as an external memory that the model can selectively draw upon based on the current context

By decoupling retrieval, selection and generation, FRAG allows for the use of task-specific retrieval and summarization models that can be optimized independently. The focused nature of the retrieved snippets helps to mitigate issues of overconfidence and extraneous information, while the notes' conciseness allows us to leave them within the context of the Interface Model for several more turns.

## Funding

FRAG's development has been funded through a research grant from [Nous Research](https://github.com/nousresearch)

## License

MIT
