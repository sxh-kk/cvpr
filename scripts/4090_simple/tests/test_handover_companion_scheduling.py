"""Live registry counts are scheduling hints, never capture admission."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collect_handover_quality100 as collector

spec=importlib.util.spec_from_file_location('companion_plan_fixture',Path(__file__).with_name('test_collect_handover_quality100.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def setup_pool(tmp_path):
    plan_path,plan=fixture.plan_file(tmp_path)
    companion=tmp_path/'companion';companion.mkdir()
    ancestors=[row['source_episode_index'] for row in plan['original_ancestors']]
    attempts=[]
    for source in ancestors:
        for variant in range(4):
            attempts.append({'status':'accepted','source_episode_index':source,'candidate_id':f'old-{source}-{variant}'})
    other={'plan_path':str(plan_path),'plan_sha256':collector.sha256(plan_path),'attempts':attempts}
    collector.write(companion/'queue.json',other)
    return plan,companion,ancestors,other


def test_pool_count_target_does_not_grant_dataset_or_policy_success(tmp_path):
    plan,companion,ancestors,_=setup_pool(tmp_path)
    own={'attempts':[{'status':'accepted','source_episode_index':source,'candidate_id':f'new-{source}'} for source in ancestors[:4]]}
    counts,hint=collector.advisory_pool(own,plan,companion)
    assert sum(counts.values())==100 and hint['selection_possible_by_count'] is True
    assert hint['dataset_ready'] is False and hint['policy_success_rate_claim'] is None


def test_pool_caps_overlapping_family_counts_but_requires_all_ancestors(tmp_path):
    plan,companion,ancestors,other=setup_pool(tmp_path)
    other['attempts']=[row for row in other['attempts'] if row['source_episode_index']!=ancestors[-1]]
    collector.write(companion/'queue.json',other)
    own={'attempts':[{'status':'accepted','source_episode_index':ancestors[0],'candidate_id':f'new-{i}'} for i in range(10)]}
    counts,hint=collector.advisory_pool(own,plan,companion)
    assert counts[ancestors[0]]==5 and hint['selection_possible_by_count'] is False


@pytest.mark.parametrize('tamper',['heldout','repeated','identity'])
def test_pool_rejects_forbidden_ancestry_duplicates_and_changed_plan(tmp_path,tamper):
    plan,companion,ancestors,other=setup_pool(tmp_path)
    own={'attempts':[]}
    if tamper=='heldout':
        other['attempts'][0]['source_episode_index']=39
    elif tamper=='repeated':
        own['attempts']=[other['attempts'][0]]
    else:
        other['plan_sha256']='0'*64
    collector.write(companion/'queue.json',other)
    with pytest.raises(ValueError):collector.advisory_pool(own,plan,companion)
