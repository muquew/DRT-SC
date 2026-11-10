from dataclasses import replace


def sender_anchor(item, worker):
    boundary = getattr(item, 'execution_boundary', None)
    if boundary is not None:
        return item.roads.normalize_position(boundary.permission_anchors[worker])
    original = getattr(item, 'original_input', item)
    return item.roads.normalize_position(original.worker_map[worker].location)


def fixed_sender_plan(item, plan):

    events = []
    for event in plan.transfers:
        senders = tuple(worker for worker, tasks in (
            (event.worker_i, event.tasks_i_to_j),
            (event.worker_j, event.tasks_j_to_i)) if tasks)
        if not senders:
            raise ValueError('handover has no sender')
        anchors = tuple(sender_anchor(item, worker) for worker in senders)
        if any(point != anchors[0] for point in anchors[1:]):
            raise ValueError('bidirectional exchange has different fixed sender positions')
        events.append(replace(event, location=anchors[0]))
    return replace(plan, transfers=tuple(events))
