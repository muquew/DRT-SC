from pathlib import Path
from common.config import routing_options as configured_routing
from common.data.roads.routing import Routing
from common.io import read_json, write_json


def resolve(root, path):
    return (Path(root)/path).expanduser().resolve()


class EpisodeStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        document = read_json(self.path)
        rows = document if isinstance(document,list) else document['episodes']
        if not isinstance(rows,list) or not rows:
            raise ValueError('DECO episodes must be a nonempty list')
        self.routing_options = document.get('routing', {}) if isinstance(document, dict) else {}
        self.episodes=[]
        seen=set()
        for row in rows:
            row=dict(row)
            if row['id'] in seen or row['split'] not in ('train','validation','learning_test'):
                raise ValueError('episode IDs must be unique and splits must be explicit')
            seen.add(row['id'])
            row['case']=str(resolve(self.path.parent,row['case']))
            row['history']=str(resolve(self.path.parent,row['history']))
            row.setdefault('distribution','ordinary')
            self.episodes.append(row)
        self.episode_routing=None
        self._loaded_path=self._loaded=None

    def rows(self, split, datasets=None):
        return [row for row in self.episodes if row['split']==split
                and (not datasets or row.get('dataset') in datasets)]

    def load(self, row):
        from common.data.pipeline import load_case
        from common.recovery.preparation import prepare_recovery_input
        from common.recovery.policy_initialization import load_history
        if self._loaded_path!=row['case']:
            raw,routing=load_case(row['case'])
            if self.routing_options:
                routing=Routing(raw.roads, **configured_routing(overrides=self.routing_options))
            prepared=prepare_recovery_input(raw,routing)
            self._loaded=prepared.projected,prepared.routing
            self._loaded_path=row['case']
        item,self.episode_routing=self._loaded
        return item,load_history(row['history'],item,require='position')


def prepare(config_path, output):
    store=EpisodeStore(config_path)
    document=dict(episodes=store.episodes)
    if store.routing_options:
        document['routing']=store.routing_options
    write_json(output,document)
    return document
