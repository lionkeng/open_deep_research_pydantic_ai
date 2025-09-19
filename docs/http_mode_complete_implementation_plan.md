# HTTP Mode API Alignment - Complete Implementation Plan

## Executive Summary

This document provides the complete implementation plan for aligning the HTTP API mode with the CLI direct mode. The plan addresses critical discrepancies between the original alignment plan's assumptions and the actual codebase implementation, incorporating best practices for Python async programming, FastAPI patterns, and production-ready architecture.

## Current State Analysis

### Identified Discrepancies

1. **Workflow Integration Mismatch**
   - **Plan Assumption**: API calls `workflow.execute_research()` method
   - **Reality**: No workflow singleton exists; class has `run()` and `resume_research()` methods
   - **Impact**: Runtime failure when API attempts to call non-existent method

2. **Session Management**
   - **Working**: `_sessions_lock` for thread safety exists
   - **Missing**: State initialization doesn't call `state.start_research()`
   - **Missing**: Proper task lifecycle management

3. **API Models**
   - **Issue**: Duplicate ResearchRequest/ResearchResponse classes in main.py
   - **Missing**: `stream_url` and `report_url` fields in responses
   - **Impact**: Model drift and validation inconsistencies

4. **Background Task Management**
   - **Current**: Uses `asyncio.create_task()` without proper tracking
   - **Missing**: Task cancellation, cleanup, and context preservation
   - **Impact**: Memory leaks, orphaned tasks, lost context

## Implementation Phases

### Phase 1: Critical Infrastructure Fixes (Days 1-2)

**Objective**: Fix breaking issues that prevent HTTP mode from functioning

#### 1.1 Workflow Integration
- Create workflow singleton in `core/workflow.py`
- Update all API method calls to use correct signatures
- Implement proper state initialization sequence

#### 1.2 API Model Alignment
- Remove duplicate model definitions
- Use canonical models from `models.api_models`
- Add missing response fields

#### 1.3 Background Task Management
- Implement `BackgroundTaskManager` class
- Add proper task lifecycle (create, track, cancel, cleanup)
- Preserve request context across async boundaries

#### 1.4 Basic Testing
- Unit tests for workflow singleton
- Integration tests for API endpoints
- Task manager lifecycle tests

### Phase 2: Robustness & State Management (Days 3-5)

**Objective**: Ensure reliable operation under concurrent load

#### 2.1 Enhanced Session Management
- Implement `ReadWriteLock` for optimal concurrent access
- Add TTL-based session cleanup
- Create `SessionRepository` abstraction for future Redis migration

#### 2.2 Two-Phase Clarification Flow
- Implement clarification pause mechanism
- Add clarification resume endpoint
- Store clarification state properly in metadata

#### 2.3 Error Handling Architecture
- Create comprehensive exception hierarchy
- Implement circuit breaker pattern
- Add structured error responses

#### 2.4 Concurrency Testing
- Load testing with multiple concurrent sessions
- Stress testing clarification flow
- Memory leak detection

### Phase 3: Production Features (Days 6-8)

**Objective**: Add enterprise-grade features for production deployment

#### 3.1 SSE Enhancements
- Implement event replay from `Last-Event-ID`
- Add backpressure handling with queue limits
- Create WebSocket fallback mechanism

#### 3.2 Observability
- Add health check endpoints
- Implement request tracing with correlation IDs
- Create metrics endpoints for monitoring

#### 3.3 Performance Optimizations
- Connection pool management for external APIs
- Event bus optimization with weak references
- Add caching layer for frequent queries

#### 3.4 Security
- Add rate limiting per session/IP
- Implement API key validation middleware
- Add request size limits

### Phase 4: Testing & Validation (Days 9-10)

**Objective**: Ensure comprehensive test coverage and quality

#### 4.1 Testing Infrastructure
- SSE stream testing utilities
- Async fixture management
- Mock event bus for testing

#### 4.2 Integration Testing
- End-to-end research flow tests
- Clarification workflow tests
- Error recovery scenarios

#### 4.3 Performance Testing
- Load testing with realistic workloads
- Memory profiling under sustained load
- Connection limit testing

#### 4.4 Documentation
- API endpoint documentation
- Deployment guide
- Troubleshooting guide

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI App                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Routers    │  │  Middleware  │  │   SSE Handler│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼────────┐   │
│  │              Background Task Manager                 │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              Session State Manager                   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              Research Workflow (Singleton)           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Singleton Pattern**: Research workflow instance
2. **Repository Pattern**: Session storage abstraction
3. **Observer Pattern**: Event bus for SSE
4. **Circuit Breaker Pattern**: Fault tolerance
5. **Factory Pattern**: Agent creation

### Async Patterns

1. **Structured Concurrency**: Using TaskGroups for related tasks
2. **Context Preservation**: Using contextvars for request context
3. **Graceful Shutdown**: Proper cleanup on application termination
4. **Backpressure Handling**: Queue limits and overflow strategies

## Python Best Practices

### Code Organization

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and routes
│   ├── task_manager.py      # Background task management
│   ├── session_manager.py   # Session state management
│   ├── middleware.py        # Custom middleware
│   ├── dependencies.py      # Dependency injection
│   └── exceptions.py        # Custom exceptions
├── core/
│   ├── workflow.py          # Research workflow with singleton
│   └── events.py            # Event bus implementation
└── models/
    └── api_models.py        # Canonical API models
```

### Type Safety

- Use Pydantic models for all API contracts
- Strict type hints throughout codebase
- Runtime validation with Pydantic validators
- Custom type guards for complex validations

### Error Handling

```python
# Exception hierarchy
ResearchException (base)
├── SessionNotFoundException (404)
├── ClarificationTimeoutException (408)
├── ResearchExecutionException (500)
├── ValidationException (422)
└── RateLimitException (429)
```

### Testing Strategy

- **Unit Tests**: 80% code coverage minimum
- **Integration Tests**: Full workflow coverage
- **Load Tests**: 100 concurrent sessions
- **SSE Tests**: Stream reliability under load

## Acceptance Criteria

### Functional Requirements

- [ ] CLI HTTP mode completes full research workflow
- [ ] Clarification flow works via HTTP endpoints
- [ ] SSE streams provide real-time updates
- [ ] Report generation and retrieval works
- [ ] Session state persists across requests
- [ ] Background tasks complete reliably

### Non-Functional Requirements

- [ ] < 200ms response time for status endpoints
- [ ] Support 100+ concurrent research sessions
- [ ] No memory leaks under 24-hour load test
- [ ] Graceful handling of client disconnections
- [ ] Proper cleanup on server shutdown
- [ ] Circuit breaker prevents cascade failures

### Quality Gates

- [ ] All tests pass: `uv run pytest`
- [ ] Code quality: `uv run ruff check --fix`
- [ ] Type checking: `uv run pyright`
- [ ] No security vulnerabilities
- [ ] Documentation complete and accurate

## Risk Mitigation

### Technical Risks

1. **Memory Leaks**
   - Mitigation: Use weak references, implement TTL cleanup
   - Monitoring: Memory profiling, metrics endpoints

2. **Task Orphaning**
   - Mitigation: Task registry, graceful shutdown
   - Monitoring: Task count metrics, cleanup logs

3. **State Corruption**
   - Mitigation: Read-write locks, atomic operations
   - Monitoring: State validation, consistency checks

4. **SSE Connection Issues**
   - Mitigation: Heartbeat mechanism, reconnection support
   - Monitoring: Connection metrics, error rates

### Operational Risks

1. **Scaling Issues**
   - Mitigation: Redis session storage option
   - Monitoring: Load metrics, response times

2. **Debugging Complexity**
   - Mitigation: Correlation IDs, structured logging
   - Monitoring: Distributed tracing, log aggregation

## Implementation Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|-------------|--------------|
| Phase 1 | 2 days | None | Working HTTP mode with basic functionality |
| Phase 2 | 3 days | Phase 1 | Robust concurrent operation |
| Phase 3 | 3 days | Phase 2 | Production-ready features |
| Phase 4 | 2 days | Phase 3 | Comprehensive testing and documentation |

## Success Metrics

- **Reliability**: 99.9% uptime for API endpoints
- **Performance**: P95 latency < 500ms for all endpoints
- **Scalability**: Support 1000+ sessions with horizontal scaling
- **Quality**: 0 critical bugs in production
- **Maintainability**: Clear documentation, 80% test coverage

## Appendix A: Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Session Management
SESSION_TTL_SECONDS=3600
SESSION_CLEANUP_INTERVAL=60
SESSION_STORAGE=memory  # or "redis"

# Redis (if used)
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=50

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# SSE Configuration
SSE_HEARTBEAT_INTERVAL=30
SSE_MAX_QUEUE_SIZE=100
SSE_RECONNECT_TIMEOUT=5
```

### Logging Configuration

```python
# Logfire configuration
logfire.configure(
    service_name="deep-research-api",
    environment="production",
    console=True,
    metrics=True,
    traces=True,
)
```

## Appendix B: API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/research` | POST | Start new research |
| `/research/{id}` | GET | Get research status |
| `/research/{id}/stream` | GET | SSE event stream |
| `/research/{id}/report` | GET | Get final report |
| `/research/{id}/clarification` | GET | Get clarification questions |
| `/research/{id}/clarification` | POST | Submit clarification response |
| `/research/{id}` | DELETE | Cancel research |

### Operational Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | OpenAPI documentation |

This implementation plan provides a comprehensive roadmap for achieving feature parity between HTTP and CLI modes while ensuring production readiness and maintainability.
