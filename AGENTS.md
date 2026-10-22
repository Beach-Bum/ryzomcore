# Ryzom engine agent entry point

This repository owns the native NeL/Ryzom engine, client, server and conversion tools. Coordination, asset contracts and task evidence live in [Beach-Bum/engine-test](https://github.com/Beach-Bum/engine-test).

Before implementation, read the coordination repository's [current AGENTS.md](https://github.com/Beach-Bum/engine-test/blob/main/AGENTS.md), the selected issue and its latest comments. The updated [production foundation](https://github.com/Beach-Bum/engine-test/pull/47) is awaiting independent review; its [pinned hub](https://github.com/Beach-Bum/engine-test/blob/517952d9c0eb0ac311b71c7812f5b21ff31f4e60/production/README.md) and [agent workflow](https://github.com/Beach-Bum/engine-test/blob/517952d9c0eb0ac311b71c7812f5b21ff31f4e60/docs/ryzom/production/agent-workflow.md) define the proposed consolidated route. Check the PR's state before describing those records as installed on main.

## Working contract

- Record the exact engine base/head, coordination issue, content revision, build/profile and intended integration branch. The current tested preparation base is `42469c82065b300ddcfba36554511698b06df337` on `work/village-height`; inspect current task evidence before choosing a base for new work.
- Claim one bounded task and reconcile any existing implementer's status. Use an isolated branch/worktree; never reset, clean, force-update or switch an occupied checkout.
- Apply the owner's versioned [research workflow](https://github.com/Beach-Bum/engine-test/blob/f56876c536e497952371e60b68c06d82140e60d6/docs/ryzom/research-workflow.md). Claude must verify and use the installed workflow; other agents must state honest manual application. Reuse settled answers; do not fabricate tool invocations.
- Preserve native client/backend behaviour, working lighting, original skeletons and existing animations. Repair the demonstrated converter/content defect before proposing replacement systems.
- Include reproducible inputs/outputs, relevant build/regression results, runtime proof and rollback. Source diagnosis, successful conversion and successful client behaviour are different results.
- Obtain substantive independent review before integration. A role or shared GitHub account cannot provide independent approval. Do not bypass failed checks or invent acceptance.
- Leave the issue handoff with revision, scope, evidence, blocker and next action; update coordination pipeline records when status changes.

## Build route

The companion change for [engine-test #46](https://github.com/Beach-Bum/engine-test/issues/46) adds PRs targeting `work/village-height` to the existing engine/client workflow filters. Existing path filters still apply; direct pushes to this branch are not newly covered. The upstream routes and build matrices remain unchanged. Default job tokens are read-only; the existing publishing job retains its own package permission and default-branch condition.

A workflow file is not proof of a successful build or an enforced merge boundary. Record actual run/artifact links; protection, independent review identity and reproducible content pairing remain tracked in [#32](https://github.com/Beach-Bum/engine-test/issues/32), [#41](https://github.com/Beach-Bum/engine-test/issues/41) and [#46](https://github.com/Beach-Bum/engine-test/issues/46).
