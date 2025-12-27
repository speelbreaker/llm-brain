# Obsidian Vault "Ops Brain" Policy

## Overview

The Obsidian vault serves as the **"Ops Brain"** - the single source of truth for operational artifacts, task management, and real-time coordination. This document defines the policies, structure, and governance rules for vault usage.

## Vault Authority Model

### The One Source of Truth Principle

**Critical Rule**: Every operational artifact has exactly ONE authoritative source. No duplicates, no conflicts, no confusion.

### Authority Assignments

**Vault is Authoritative For**:
- Task queues and prioritization (02_QUEUE/)
- Current priorities and focus (01_NOW/)
- Operational procedures and runbooks (04_OPS/)
- Incident reports and response procedures (04_OPS/)
- Prompt library and templates (06_PROMPTS/)
- Agent coordination workflows (06_PROMPTS/)
- Real-time system status (04_OPS/)

**Repository is Authoritative For**:
- Code and code changes (git repository)
- Technical documentation (docs/ directory)
- Architecture decisionsdocs/architecture (/)
- API specifications and contracts (docs/contracts/)
- Development workflows and procedures
- Version control and change history

## Vault Structure

### Required Directory Structure

```
docs/obsidian/
├── 00_README.md              # Vault overview and navigation
├── 01_NOW/                   # Current priorities and focus
│   ├── NOW.md                # Current sprint/objectives
│   ├── BLOCKERS.md           # Current impediments
│   └── PRIORITIES.md         # Priority queue
├── 02_QUEUE/                 # Task queue (machine-parsable)
│   ├── QUEUE.md              # Main task queue
│   ├── EMERGENCY.md          # High-priority tasks
│   └── COMPLETED/            # Archive of completed tasks
├── 03_LOGS/                  # System and decision logs
│   ├── CHANGELOG.md          # System changes
│   ├── supervisor_runs.md    # Supervisor execution log
│   └── DECISIONS.md          # Decision audit trail
├── 04_OPS/                   # Operations procedures
│   ├── INCIDENT_LOG.md       # Incident reports
│   ├── RUNBOOKS/             # Operational procedures
│   └── MONITORING/           # System health and alerts
├── 06_PROMPTS/               # Agent prompt library
│   ├── _ACTIVE.md            # Currently active prompt
│   ├── SUPERVISOR/           # Supervisor-specific prompts
│   ├── AGENTS/               # Agent coordination prompts
│   └── TEMPLATES/            # Reusable prompt templates
└── 99_ARCHIVE/               # Historical data
    ├── PROMPTS/              # Old prompts
    ├── TASKS/                # Completed tasks
    └── INCIDENTS/            # Resolved incidents
```

### Queue File Format (Machine-Parsable)

**QUEUE.md Structure**:
```markdown
# Task Queue

Last updated: 2025-01-27T01:27:51Z

## READY
- TASK-001 | Fix security vulnerability in runner
- TASK-002 | Update documentation for MCP integration

## IN_PROGRESS  
- TASK-003 | Implement queue manager tests

## IN_REVIEW
- TASK-004 | Review decision loop contract

## DONE
- TASK-005 | Create secure runner implementation
```

**Queue Item Format**:
```
- <TASK_ID> | <Description> | key: value | key: value
```

**Required Fields**:
- `TASK_ID`: Unique identifier (format: TASK-###)
- `Description`: Human-readable task description
- `branch`: Git branch name (if applicable)
- `prompt`: Path to prompt file (if applicable)
- `agent`: Assigned agent name
- `started`: ISO timestamp when work began
- `pr`: Pull request URL (if applicable)
- `updated`: Last modification timestamp
- `completed`: Completion timestamp

## Queue Discipline Rules

### No Pile-Ups Policy

**Maximum Queue Depth**:
- READY: Unlimited (managed by priority)
- IN_PROGRESS: Maximum 1 item per priority level
- IN_REVIEW: Maximum 1 item
- DONE: Unlimited (archived items)

**Queue Position Enforcement**:
- Strict FIFO within priority levels
- Priority inversion forbidden
- No queue jumping without explicit approval

### State Transition Rules

**READY → IN_PROGRESS**:
- Agent claims task and creates branch
- Updates task with `agent`, `started`, `branch` fields
- Moves to IN_PROGRESS section

**IN_PROGRESS → IN_REVIEW**:
- Work completed and tested
- Pull request created
- Updates task with `pr` URL
- Moves to IN_REVIEW section

**IN_REVIEW → DONE**:
- Code reviewed and merged
- Updates task with `completed` timestamp
- Moves to DONE section

## Mirror Synchronization

### Automated Sync (Dev → Ops)

**Triggered By**:
- Code deployment to production
- Documentation updates
- Architecture decision changes

**Sync Process**:
1. Repository changes detected
2. Automated sync job triggered
3. Changes mirrored to vault
4. Sync confirmation logged

**Forbidden Items** (Never Sync):
- Secrets, API keys, or credentials
- Internal implementation details
- Development-only artifacts

### Manual Review Required (Ops → Dev)

**Process**:
1. Operational change identified
2. Impact assessment completed
3. Manual review by technical team
4. If approved: Update repository documentation
5. Sync confirmation logged

## Agent Coordination

### Supervisor Integration

**Queue File**: `02_QUEUE/QUEUE.md`
- Machine-parsable format for supervisor
- Atomic updates with backup/recovery
- Real-time synchronization

**Active Prompt**: `06_PROMPTS/_ACTIVE.md`
- Current supervisor prompt location
- Updated by agents as needed
- Tracked for audit purposes

### Human Operator Interface

**Emergency Procedures**: `04_OPS/INCIDENT_LOG.md`
- Incident report format
- Response procedures
- Escalation contacts

**Status Dashboard**: Real-time vault view
- Current queue state
- Active tasks and blockers
- System health indicators

## Governance & Compliance

### Access Control

**Read Access**:
- All team members: Full read access
- External stakeholders: Limited read access
- Automated systems: Service account access

**Write Access**:
- Human operators: Manual editing
- Supervisors: Automated queue updates
- Agents: Prompt library updates only

**Administrative Access**:
- Vault administrators: Full control
- Emergency access: Incident response team
- Regular review: Quarterly access audit

### Change Management

**Routine Changes**:
- Queue updates: Real-time
- Documentation: Review before commit
- Procedures: Version controlled

**Emergency Changes**:
- Incident procedures: Immediate
- System status: Real-time
- Escalation: Immediate notification

### Audit Requirements

**Mandatory Logging**:
- All queue state changes
- Prompt library modifications  
- Emergency procedure activations
- Cross-system synchronizations

**Retention Policy**:
- Queue history: 1 year minimum
- Incident reports: 7 years minimum
- Decision logs: 7 years minimum
- Prompt evolution: Permanent record

## Quality Assurance

### Data Integrity

**Queue Validation**:
- Machine-readable format verification
- Required field completeness checks
- Duplicate detection and prevention
- Atomic update procedures

**Content Standards**:
- Consistent formatting
- Clear, actionable descriptions
- Proper categorization and tagging
- Regular archival of completed items

### Monitoring & Alerting

**Health Checks**:
- Queue file accessibility
- Sync job status
- Backup integrity
- Access control compliance

**Alert Conditions**:
- Queue corruption detected
- Sync failures
- Unauthorized access attempts
- Backup creation failures

## Migration & Backup

### Backup Strategy

**Automated Backups**:
- Daily full vault backup
- Real-time queue file backup
- Cross-system redundancy
- Cloud storage synchronization

**Recovery Procedures**:
- Automated recovery from latest backup
- Manual intervention for complex issues
- Verification procedures post-recovery
- Incident reporting for data loss

### Migration Planning

**Vault Structure Changes**:
- Backward compatibility requirements
- Migration scripts and procedures
- Validation checkpoints
- Rollback capabilities

**System Integration Changes**:
- Supervisor compatibility
- Agent coordination updates
- Mirror synchronization changes
- API compatibility maintenance

## Success Metrics

### Operational Effectiveness

**Queue Management**:
- Average task completion time: Target <48 hours
- Queue depth management: Alert if >20 pending
- State transition accuracy: 100%

**Data Quality**:
- Format compliance: >99%
- Required field completeness: 100%
- Sync success rate: >99.9%

### User Experience

**Operator Efficiency**:
- Time to find information: <30 seconds
- Queue update accuracy: >99%
- Emergency procedure access: <10 seconds

**System Reliability**:
- Vault availability: >99.9%
- Sync success rate: >99.9%
- Backup integrity: 100%

This policy ensures the Obsidian vault serves as a reliable, authoritative source of operational truth while maintaining security, integrity, and operational effectiveness.
