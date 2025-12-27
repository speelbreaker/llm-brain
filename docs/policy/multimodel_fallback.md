# Multi-Model Fallback Policy

## Overview

This document defines the fallback strategy for AI model providers across different roles in the decision loop architecture. The goal is to ensure continuous operation while maintaining quality and cost efficiency through intelligent provider selection and failure handling.

## Provider Hierarchy by Role

### Optimist Agent

**Primary Provider**: OpenAI GPT-4o
- Strengths: Creative thinking, market analysis, opportunity identification
- Best for: Generating optimistic but evidence-based proposals
- Context window: 128K tokens
- Rate limits: 10K tokens/minute

**Secondary Provider**: Anthropic Claude
- Strengths: Detailed analysis, logical reasoning, risk consideration
- Best for: Balanced proposals with thorough evidence
- Context window: 200K tokens
- Rate limits: 8K tokens/minute

**Tertiary Provider**: Local Model (Llama/Mistral)
- Strengths: Cost-effective, privacy-preserving
- Best for: Simple proposals with clear requirements
- Context window: 32K tokens
- Rate limits: Unlimited (local processing)

**Fallback Sequence**: OpenAI → Anthropic → Local Model → Degrade-to-safe-NO_ACTION

### Skeptic Agent

**Primary Provider**: Anthropic Claude
- Strengths: Critical thinking, risk assessment, logical challenges
- Best for: Identifying flaws and potential risks
- Context window: 200K tokens
- Rate limits: 8K tokens/minute

**Secondary Provider**: OpenAI GPT-4
- Strengths: Analytical thinking, pattern recognition
- Best for: Technical risk assessment and validation
- Context window: 128K tokens
- Rate limits: 10K tokens/minute

**Tertiary Provider**: Local Model (Llama/Mistral)
- Strengths: Rule-based validation, pattern matching
- Best for: Basic risk screening
- Context window: 32K tokens
- Rate limits: Unlimited

**Fallback Sequence**: Anthropic → OpenAI → Local Model → Degrade-to-safe-NO_ACTION

### Arbiter Agent

**Primary Provider**: OpenAI GPT-4o
- Strengths: Decision-making, weighing trade-offs, final judgment
- Best for: Making final go/no-go decisions
- Context window: 128K tokens
- Rate limits: 10K tokens/minute

**Secondary Provider**: Google Gemini
- Strengths: Multi-modal reasoning, structured decision-making
- Best for: Complex decision trees with conditions
- Context window: 1M tokens
- Rate limits: 15K tokens/minute

**Tertiary Provider**: Local Model (Llama/Mistral)
- Strengths: Rule-based decisions, consistent logic
- Best for: Simple binary decisions
- Context window: 32K tokens
- Rate limits: Unlimited

**Fallback Sequence**: OpenAI → Google Gemini → Local Model → Degrade-to-safe-NO_ACTION

### Executor Agent

**Primary Provider**: Anthropic Claude
- Strengths: Code generation, implementation planning, technical execution
- Best for: Technical implementation and execution planning
- Context window: 200K tokens
- Rate limits: 8K tokens/minute

**Secondary Provider**: OpenAI GPT-4
- Strengths: Technical writing, API integration, tool usage
- Best for: API calls, system integration, execution monitoring
- Context window: 128K tokens
- Rate limits: 10K tokens/minute

**Tertiary Provider**: Local Model (Llama/Mistral)
- Strengths: Simple command execution, basic automation
- Best for: Straightforward execution tasks
- Context window: 32K tokens
- Rate limits: Unlimited

**Fallback Sequence**: Anthropic → OpenAI → Local Model → Degrade-to-safe-NO_ACTION

## Fallback Trigger Conditions

### Timeout Triggers

**Primary Provider Timeout**:
- Timeout threshold: 30 seconds
- Retry attempts: 2 (with exponential backoff)
- Total timeout budget: 90 seconds
- Trigger: If no response within timeout

**Secondary Provider Timeout**:
- Timeout threshold: 45 seconds (longer due to potential load)
- Retry attempts: 1
- Total timeout budget: 90 seconds
- Trigger: If primary consistently times out

### Error Rate Triggers

**Provider Error Rate**:
- Alert threshold: >5% error rate over 10 requests
- Failure threshold: >10% error rate over 10 requests
- Rolling window: Last 20 requests
- Trigger: If error rate exceeds thresholds

**Consecutive Failure Limit**:
- Limit: 3 consecutive failures
- Reset condition: 5 successful requests
- Trigger: After 3 consecutive timeouts or errors

### Rate Limit Triggers

**Quota Consumption**:
- Warning threshold: >80% of daily quota
- Critical threshold: >95% of daily quota
- Reset period: Daily at midnight UTC
- Trigger: Quota exhaustion imminent

**Rate Limiting Events**:
- Immediate trigger: Rate limit response received
- Cool-down period: 60 seconds
- Notification: Operator alert for repeated events

### Quality Degradation Triggers

**Response Quality Assessment**:
- Minimum score: 3.0/5.0 on quality rubric
- Scoring criteria: Coherence, relevance, completeness
- Trigger: Below minimum score for 3 consecutive responses

**Consistency Checks**:
- Decision consistency: <80% agreement across multiple evaluations
- Pattern recognition: Inconsistent recommendations for similar inputs
- Trigger: Quality inconsistency detected

## Circuit Breaker Behavior

### Circuit Breaker States

**CLOSED (Normal Operation)**:
- All providers accepting requests
- Normal fallback behavior
- Monitoring and alerting active

**OPEN (Failure State)**:
- Failed provider blocked from requests
- Requests routed to healthy providers
- Timeout and error rate monitoring
- Automatic retry after cool-down

**HALF-OPEN (Testing State)**:
- Limited request volume to test provider
- Success rate monitoring
- Return to CLOSED if healthy
- Return to OPEN if failures continue

### Provider-Specific Circuit Breakers

**OpenAI Circuit Breaker**:
- Trigger: 3 consecutive failures or >10% error rate
- Cool-down: 60 seconds
- Test requests: 5 (must all succeed to close)
- Max open time: 15 minutes

**Anthropic Circuit Breaker**:
- Trigger: 3 consecutive failures or timeout rate >15%
- Cool-down: 90 seconds (more conservative)
- Test requests: 3 (must all succeed to close)
- Max open time: 20 minutes

**Google Gemini Circuit Breaker**:
- Trigger: 2 consecutive failures (newer provider, more conservative)
- Cool-down: 45 seconds
- Test requests: 3
- Max open time: 10 minutes

**Local Model Circuit Breaker**:
- Trigger: System resource exhaustion
- Cool-down: Manual reset required
- Test requests: System health check
- Max open time: Until manual intervention

### Global Circuit Breaker

**System-Wide Circuit Breaker**:
- Trigger: All external providers failing
- Action: Degrade-to-safe-NO_ACTION mode
- Cooldown: 5 minutes
- Recovery: Manual operator review required

**Domain-Specific Circuit Breakers**:
- Trading domain: More aggressive circuit breaking
- Coding domain: More tolerant of temporary failures
- Operations: Balanced approach with safety margins

## Degrade-to-Safe-NO_ACTION Mode

### Default Behavior

**When All Providers Fail**:
- Immediate switch to safe-NO_ACTION mode
- No new decisions or implementations initiated
- Queue all requests for later processing
- Notify operators within 2 minutes

**Safe Defaults by Domain**:

**Trading Domain**:
- No new positions opened
- Existing positions maintained
- No leveraged trades
- Conservative risk management
- Emergency stop if high-risk conditions

**Coding Domain**:
- No production deployments
- No database changes
- No security-sensitive schema modifications
- Safe rollbacks only
- Monitoring and alerting only

**Operations Domain**:
- No infrastructure changes
- No configuration modifications
- No user access changes
- Emergency procedures only
- Status monitoring only

### Recovery Procedures

**Manual Recovery Required**:
- All providers must be tested individually
- System health must be verified
- Operator approval for resumption
- Gradual ramp-up from 25% capacity

**Automatic Recovery**:
- Provider health checks every 5 minutes
- Gradual capacity increase: 25% → 50% → 75% → 100%
- Continuous monitoring during recovery
- Automatic revert to safe mode if issues detected

## Data Consistency Invariant

### Market Data Consistency

**Single Snapshot Rule**:
- All agents in a decision loop must use the same market data snapshot
- Snapshot timestamp must be <5 minutes old
- No partial updates during decision processing
- Rollback to previous snapshot if inconsistency detected

**Data Validation**:
- Price sanity checks (±50% deviation triggers alert)
- Volume verification against historical patterns
- Liquidity assessment for trading decisions
- Timestamp validation and ordering

### Repository Context Consistency

**Repository State Rule**:
- All agents analyzing code must use same repository state
- Snapshot includes: branch, commit hash, changed files
- No concurrent modifications during decision processing
- Validation against repository metadata

**Change Detection**:
- Monitor for concurrent commits
- Alert on unexpected repository changes
- Pause decisions if state changes during processing
- Resume with updated snapshot after validation

### Queue State Consistency

**Queue Integrity Rule**:
- Queue state must be atomic and consistent
- No partial updates during decision processing
- Backup and recovery procedures mandatory
- Validation of queue format and content

## Monitoring & Alerting

### Provider Health Monitoring

**Real-Time Metrics**:
- Response time per provider
- Success/error rates
- Quality scores
- Rate limit usage

**Alert Thresholds**:
- Response time >30 seconds: Warning
- Error rate >5%: Warning
- Error rate >10%: Critical
- Provider timeout >2 minutes: Critical

### Fallback Event Tracking

**Fallback Occurrence**:
- Count of fallback events per provider
- Reason for fallback (timeout, error, quality)
- Time spent in fallback mode
- Impact on decision loop performance

**Quality Impact Assessment**:
- Compare decision quality across providers
- Identify patterns in provider performance
- Track operator intervention frequency
- Monitor decision consistency

### Cost Optimization

**Provider Cost Tracking**:
- Cost per token by provider
- Total daily/weekly/monthly costs
- Cost per successful decision
- ROI analysis by provider role

**Optimization Opportunities**:
- Identify over-provisioned capacity
- Suggest provider rebalancing
- Recommend contract renegotiations
- Track cost vs. quality trade-offs

## Performance Benchmarks

### Response Time Targets

**By Provider Type**:
- OpenAI GPT-4o: <15 seconds
- Anthropic Claude: <20 seconds
- Google Gemini: <12 seconds
- Local Model: <5 seconds

**By Decision Phase**:
- Optimist: <20 seconds
- Skeptic: <25 seconds
- Arbiter: <15 seconds
- Executor: <30 seconds (includes implementation time)

### Quality Benchmarks

**Decision Quality Metrics**:
- Consistency: >90% agreement across multiple evaluations
- Completeness: >95% of required elements present
- Relevance: >90% relevance score from human evaluators
- Actionability: >85% decisions implementable without modification

**Provider-Specific Targets**:
- OpenAI: >4.0/5.0 average quality score
- Anthropic: >4.2/5.0 average quality score
- Google: >3.8/5.0 average quality score
- Local: >3.5/5.0 average quality score

### Reliability Targets

**Availability Targets**:
- OpenAI: >99.5% uptime
- Anthropic: >99.0% uptime
- Google: >99.0% uptime
- Local: >99.9% uptime

**Fallback Performance**:
- Fallback success rate: >98%
- Average fallback time: <30 seconds
- Quality preservation: >90% of original quality
- Operator intervention: <5% of fallback events

## Implementation Guidelines

### Configuration Management

**Provider Configuration**:
- API keys and endpoints stored securely
- Rate limits and timeouts configurable
- Quality thresholds adjustable per role
- Circuit breaker settings tunable

**Environment-Specific Settings**:
- Development: Lower thresholds, more aggressive fallbacks
- Staging: Production-like settings with monitoring
- Production: Conservative thresholds, careful fallbacks

### Testing Requirements

**Provider Testing**:
- Automated health checks every hour
- Response quality testing weekly
- Cost optimization testing monthly
- Security assessment quarterly

**Fallback Testing**:
- Simulated provider failures
- Circuit breaker validation
- Recovery procedure testing
- Performance impact assessment

### Operator Training

**Fallback Recognition**:
- Identify fallback events in logs
- Understand quality implications
- Know when manual intervention required
- Recovery procedure familiarity

**Troubleshooting Skills**:
- Provider-specific issues identification
- Configuration problem resolution
- Performance optimization techniques
- Emergency recovery procedures

This multi-model fallback policy ensures resilient, high-quality decision-making while optimizing for cost, performance, and reliability across all operational domains.
