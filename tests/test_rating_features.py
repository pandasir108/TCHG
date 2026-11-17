from copy import deepcopy

import numpy as np
import pytest

from tchg.data.rating_features import causal_rating_features, load_causal_rating_features


def examples():
    query={'src':np.zeros(4,dtype=np.int64),'dst':np.ones(4,dtype=np.int64),
           'timestamp':np.array([2.,3.,4.,6.]),'split':np.array([0,0,1,2]),'num_nodes':np.array([2]),
           'label':np.ones(4),'observed_mask':np.ones(4)}
    ratings=np.array([[0,10,1,5,1,1],[1,10,1,4,1,2],
                      [1,20,2,2,1,3],[0,20,2,1,1,5]],dtype=float)
    return query,ratings


def test_rating_history_excludes_simultaneous_events_and_freezes_evaluation_splits():
    query,ratings=examples();x=causal_rating_features(query,ratings)
    np.testing.assert_allclose(x['entity_features'][0,:2],np.log1p([1,0])/10.)
    assert x['behavior_features'][0,0]==0
    np.testing.assert_allclose(x['behavior_features'][1],[np.log(2)/10.,1.,.75])
    np.testing.assert_allclose(x['entity_features'][2,:2],np.log1p([1,2])/10.)
    np.testing.assert_array_equal(x['entity_features'][2],x['entity_features'][3])
    streaming=causal_rating_features(query,ratings,history_policy='pre_event')
    np.testing.assert_allclose(streaming['entity_features'][3,:2],np.log1p([2,2])/10.)
    assert streaming['behavior_features'][3,1]==1.


def test_future_ratings_and_trust_labels_or_masks_cannot_change_rating_evidence():
    query,ratings=examples();before=causal_rating_features(query,ratings)
    altered=deepcopy(query);altered['label'][:]=0;altered['observed_mask'][:]=0
    extended=np.vstack([ratings,[0,20,2,5,5,100]])
    after=causal_rating_features(altered,extended)
    for key in before:np.testing.assert_array_equal(before[key],after[key])


def test_recent_profiles_expire_and_do_not_depend_on_future_ratings():
    query,ratings=examples()
    query['timestamp']*=86400.;ratings[:,5]*=86400.
    first=causal_rating_features(query,ratings,'pre_event',profile_dim=16,profile_window_days=2.,temporal_activity=True)
    assert first['rating_profile_src'].shape==(4,16)
    np.testing.assert_array_equal(first['rating_profile_src'][2],0.)
    assert first['rating_profile_src'][3].any()
    more=np.vstack([ratings,[0,30,3,5,1,100*86400.]])
    after=causal_rating_features(query,more,'pre_event',profile_dim=16,profile_window_days=2.,temporal_activity=True)
    for key in first:np.testing.assert_array_equal(first[key],after[key])
    assert first['entity_features'].shape[1]==16 and first['context_features'].shape[1]==4


def test_rating_loader_uses_raw_user_ids_and_cache_ignores_trust_query_labels(tmp_path):
    folder=tmp_path/'epinions';folder.mkdir()
    (folder/'user_user_time.txt').write_text('10 20 1\n20 10 3\n')
    (folder/'user_item_time.txt').write_text('10,100,1,5,1,0,title,with,commas\n20,100,1,4,1,1,another title\n')
    query={'src':np.array([0,0]),'dst':np.array([1,1]),'timestamp':np.array([2.,4.]),
           'split':np.array([0,2]),'num_nodes':np.array([2]),'node_to_raw':np.array([10,20]),
           'label':np.ones(2),'observed_mask':np.ones(2)}
    x,meta=load_causal_rating_features(query,tmp_path,tmp_path/'cache.npz')
    np.testing.assert_allclose(x['entity_features'][0,:2],np.log1p([1,1])/10.)
    assert x['behavior_features'][0,1]==1.
    assert not meta['trust_labels_used']
    query['label'][:]=0;query['observed_mask'][:]=0
    cached,_=load_causal_rating_features(query,tmp_path,tmp_path/'cache.npz')
    for key in x:np.testing.assert_array_equal(x[key],cached[key])
    query['node_to_raw']=np.array([20,10])
    with pytest.raises(ValueError,match='node_to_raw'):
        load_causal_rating_features(query,tmp_path,tmp_path/'cache.npz')
