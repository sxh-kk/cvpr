"""XMoveBendPick-only durable initialization progress; Mobile defaults stay unchanged."""
from pico_mobile_collection_progress import CollectionProgress as _Progress
from pico_xmove_bend_pick_collection_plan import EPISODE_INDICES, TASK_ID


class CollectionProgress(_Progress):
    contract = 'cvpr-xmove_bend_pick-official100-collection-progress-v1'
    acceptance_gate_version = 'xmove_bend_pick-native-task-success-v1'

    def __init__(self, rows, path):
        if (len(rows) != 100 or [row.get('official_episode_index') for row in rows] != EPISODE_INDICES
                or any(row.get('task_id') != TASK_ID for row in rows)):
            raise ValueError('XMoveBendPick progress requires its exact official100 plan')
        super().__init__(rows, path)
