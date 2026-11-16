import numpy as np

from tchg.data.structural_features import causal_structural_features


def events():
    return {'src':np.array([0,1,0,1,2,2,3,0]),
            'dst':np.array([1,2,2,0,0,3,2,1]),
            'timestamp':np.array([1.,1.,2.,3.,4.,5.,6.,7.]),
            'split':np.array([0,0,0,1,1,2,2,2]),
            'sample_type':np.array([0,0,1,0,0,0,0,0]),
            'label':np.array([1.,1.,0.,1.,1.,1.,1.,1.]),
            'observed_mask':np.ones(8),'num_nodes':np.array([4])}


def test_structure_uses_strict_past_and_replays_observed_validation():
    a=events();x=causal_structural_features(a)
    np.testing.assert_array_equal(x[:2],0.)
    assert x[2,2]>0  # historical 0 -> 1 -> 2
    assert x[3,0]==1  # historical 0 -> 1, querying 1 -> 0
    assert x[4,0]==0  # negative candidate 0 -> 2 was not written
    assert x[6,0]==0  # earlier test query 2 -> 3 was not written
    assert x[7,0]==1  # observed validation 1 -> 0 is now historical


def test_structure_does_not_consult_test_labels_or_observation_masks():
    a=events();x=causal_structural_features(a)
    a['label'][a['split']==2]=0.
    a['sample_type'][a['split']==2]=1
    a['observed_mask'][a['split']==2]=0.
    np.testing.assert_array_equal(x,causal_structural_features(a))


def test_missing_validation_evidence_does_not_enter_test_graph():
    a=events();a['observed_mask'][3]=0.
    assert causal_structural_features(a)[7,0]==0


def test_structural_cache_is_invalidated_when_past_visibility_changes(tmp_path):
    a=events();p=tmp_path/'structure.npz'
    x=causal_structural_features(a,p)
    np.testing.assert_array_equal(x,causal_structural_features(a,p))
    a['observed_mask'][3]=0.
    assert causal_structural_features(a,p)[7,0]!=x[7,0]


def test_simultaneous_edges_and_future_evidence_do_not_change_earlier_queries():
    a=events()
    a['dst'][1]=0  # 0 -> 1 and 1 -> 0 occur simultaneously.
    x=causal_structural_features(a)
    assert x[0,0]==x[1,0]==0
    a['src'][3:]=3
    a['dst'][3:]=0
    a['label'][3:]=0.
    np.testing.assert_array_equal(x[:3],causal_structural_features(a)[:3])
