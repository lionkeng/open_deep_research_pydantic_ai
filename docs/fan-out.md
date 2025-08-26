````mermaid
flowchart TD
    U[User Query] -->|optional Q&A| C[Clarifier]
    C -->|finalized brief| O[Orchestrator]

    subgraph PLAN[Decomposition & Search Planning]
      O --> P[GoT Planner - sub-topics as graph nodes]
      P --> M[MCTS Controller - LATS-style UCB1: select → expand → simulate → backprop]
      M -->|priority queue of nodes| Q[Executor Pool]
    end

    subgraph EXEC[Execution Fan-out]
      Q -->|parallel| T1[Tool Calls - web search, APIs, papers]
      Q -->|parallel| T2[Retrieval Augmenter - RAG over corpora]
      Q -->|parallel| T3[Verification Tools - critics, calculators, code, cite-check]
    end

    subgraph EVIDENCE[Evidence & Memory]
      T1 --> S[(Evidence Store - Docs, snippets, metadata, embeddings)]
      T2 --> S
      T3 --> S
      S --> D[Dedup/Normalize/Chunk - source credibility scoring]
    end

    subgraph AGG[Aggregation & Judgment]
      D --> A[Subtopic Summarizers - per node]
      A --> MOA[Mixture-of-Agents Consensus - cross-summaries + judge]
      MOA --> R1[Integrated Draft]
      R1 --> RF[Self-Refine / Reflexion - critic → revise loops]
      RF --> REP[Report Generator - outline → sections → citations]
    end

    REP --> OUT[(Final Research Report - citations, appendix, sources)]
    ```

    ```mermaid
    sequenceDiagram
    participant U as User
    participant C as Clarifier
    participant O as Orchestrator
    participant P as GoT Planner
    participant M as MCTS Controller
    participant X as Executor Pool
    participant S as Evidence Store
    participant J as MoA Judge
    participant F as Self-Refine
    participant G as Report Gen

    U->>C: Free-form topic
    C-->>U: (optional) targeted clarifying questions
    C->>O: Finalized research brief (scope, constraints, KPIs)
    O->>P: Decompose into subtopics (graph nodes)
    P->>M: Initial frontier (priority by impact/uncertainty)
    loop Search iterations (budgeted)
      M->>X: Select K nodes to expand (UCB1)
      X->>S: Fan-out: search/APIs/RAG/verification → store evidence
      M->>M: Backprop scores (coverage, credibility, novelty, utility)
    end
    O->>J: Per-node summaries + cross-reading
    J->>F: Consensus draft with cites
    F->>F: Critique → Revise (N passes or until quality threshold)
    F->>G: Structured report (outline → sections)
    G-->>U: Final report + sources + appendix
````
