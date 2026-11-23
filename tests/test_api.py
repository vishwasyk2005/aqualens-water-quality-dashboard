"""Service and real HTTP contract tests for the local custom dashboard."""
import http.client
import gzip
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Dashboard, APIError, clean, make_server, MAX_BODY, Handler
from aqualens.data import load_demo

class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=Dashboard()

    def test_json_clean_preserves_missing_and_dates(self):
        result=clean({'nan':np.nan,'inf':float('inf'),'n':np.int64(3),'date':pd.Timestamp('2025-01-01'),'missing':pd.NA,'missing_date':pd.NaT})
        self.assertEqual(result['nan'],None)
        self.assertEqual(result['inf'],None)
        self.assertEqual(result['date'],'2025-01-01T00:00:00')
        self.assertIsNone(result['missing_date'])
        json.dumps(result,allow_nan=False)

    def test_large_json_compresses_only_when_client_accepts_gzip(self):
        handler=Handler.__new__(Handler)
        handler.headers={'Accept-Encoding':'gzip, deflate'}
        handler.wfile=io.BytesIO()
        content={'rows':['observation']*5000}
        with patch.object(handler,'response_headers') as headers:
            handler.json(content)
        self.assertEqual(headers.call_args.args[-1],'gzip')
        self.assertEqual(json.loads(gzip.decompress(handler.wfile.getvalue())),content)
        handler.headers={'Accept-Encoding':'gzip;q=0'}
        handler.wfile=io.BytesIO()
        with patch.object(handler,'response_headers') as headers:
            handler.json(content)
        self.assertIsNone(headers.call_args.args[-1])
        self.assertEqual(json.loads(handler.wfile.getvalue()),content)
        with patch.object(handler,'send_response'),patch.object(handler,'send_header') as header,patch.object(handler,'end_headers'):
            handler.response_headers(200,'application/json',100,'gzip')
        header.assert_any_call('Content-Encoding','gzip')
        header.assert_any_call('Vary','Accept-Encoding')

    def test_bootstrap_is_complete_and_does_not_train(self):
        with patch('server.train_models',side_effect=AssertionError('Bootstrap must not train')):
            result=self.app.bootstrap()
        self.assertEqual(len(result['observations']),result['report']['rows'])
        self.assertEqual(len(result['zones']),6)
        self.assertEqual(len(result['alerts']),18)
        self.assertEqual({m['id'] for m in result['models']},{'det','auto','prophet'})
        self.assertIn('seasonal_kernel', {model['id'] for model in result['model_benchmark']['models']})
        self.assertTrue(result['knowledge'])
        self.assertIn('condition',result)
        self.assertEqual(result['report']['source'],'legacy_upload')
        self.assertEqual(result['condition']['as_of'][:10],result['report']['end'])
        json.dumps(result,allow_nan=False)

    def test_bootstrap_reuses_scientific_result_for_same_checkpoint(self):
        first=self.app.bootstrap()
        with patch('server.current_condition',side_effect=AssertionError('Already cached')):
            second=self.app.bootstrap()
        self.assertIs(first,second)

    def test_modified_data_cannot_reuse_demo_checkpoint(self):
        frame=self.app.demo[0].copy()
        frame.loc[0,'turbidity_proxy']+=0.01
        options=self.app.models(frame)
        self.assertFalse(next(m['available'] for m in options if m['id']=='det'))
        self.assertEqual(len(options),1)

    def test_det_forecast_contract_and_no_fallback(self):
        app=Dashboard()
        with patch.object(app,'checkpoint',return_value=(None,{})):
            with self.assertRaises(APIError) as error:
                app.forecast('modis','det',32)
            self.assertEqual(error.exception.status,409)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'work'/'train-temp') as tmp:
            checkpoint=Path(tmp)/'det_v2.pt';checkpoint.write_bytes(b'test')
            rows=pd.DataFrame([dict(date='2023-02-11',zone_id='Z1',parameter='ndwi',prediction=.2,lower=.1,upper=.3,model='det',horizon_days=1,is_extrapolation=False)])
            with patch.object(app,'checkpoint',return_value=(checkpoint,{'version':2})),patch.object(app,'has_torch',return_value=True),patch('server.forecast_zones',return_value=rows) as predictor:
                result=app.forecast('modis','det',1)
                self.assertEqual(result['model'],'det')
                self.assertEqual(result['rows'][0]['prediction'],.2)
                self.assertEqual(result['metadata']['max_validated_horizon_days'],32)
                self.assertEqual(predictor.call_args.kwargs['horizon_days'],1)
                self.assertEqual(predictor.call_args.kwargs['model'],'det')

    def test_future_date_uses_latest_uploaded_date(self):
        rows=[dict(date='2023-02-11',zone_id='Z1',parameter='ndwi',prediction=.3)]
        with patch.object(self.app,'forecast',return_value=dict(rows=rows,metadata={})) as forecast:
            result=self.app.predict_date('modis','2023-02-11')
            self.assertEqual(result['horizon_days'],1)
            self.assertEqual(result['rows'],rows)
            forecast.assert_called_once_with('modis','det',1)
        for target in ['invalid','2023-02-10','2020-01-01']:
            with self.assertRaises(APIError):self.app.predict_date('modis',target)

    def test_invalid_model_horizon_zone_rejected(self):
        for model,horizon in [('mean_pool',32),('tree',32),('det',0),('det',731),('seasonal_kernel',731),('prophet',731)]:
            with self.assertRaises(APIError): self.app.forecast('demo',model,horizon)
        with self.assertRaises(APIError): self.app.explain('demo','missing','ndwi')

    def test_upload_validate_register_and_bound_retention(self):
        app=Dashboard()
        csv=load_demo()[0].to_csv(index=False)
        with patch.object(app,'bootstrap',side_effect=lambda key:{'dataset':key}):
            ids=[app.upload({'csv':csv,'filename':'observations.csv'})['dataset'] for _ in range(6)]
        self.assertEqual(len(app.datasets),5)
        with self.assertRaises(APIError): app.dataset(ids[0])
        self.assertEqual(app.dataset(ids[-1])[1]['filename'],'observations.csv')
        with self.assertRaises(ValueError): app.upload({'csv':'bad,columns\n1,2'})

    def test_evidence_collection_isolation(self):
        app=Dashboard()
        record=dict(source_id='custom-one',title='Rainfall runoff',url='https://example.org/report',text='Rainfall runoff suspended sediment.',applicability='Test passage',parameters=['turbidity_proxy'])
        result=app.add_knowledge({'records':[record]})
        self.assertEqual(result['count'],1)
        self.assertEqual(app.knowledge(result['evidence_id'])[0]['source_id'],'custom-one')
        self.assertNotIn('custom-one',{r['source_id'] for r in app.knowledge()})
        hits=app.get('/api/evidence',{'q':['runoff'],'evidence_id':[result['evidence_id']]})
        self.assertEqual(hits['results'][0]['source_id'],'custom-one')
        with self.assertRaises(ValueError): app.add_knowledge({'records':[dict(record,url='javascript:alert(1)')]})

    def test_explanation_provenance(self):
        result=self.app.explain('demo','Z1','turbidity_proxy')
        self.assertEqual(result['provenance']['zone_id'],'Z1')
        self.assertEqual(len(result['hypotheses']),4)
        self.assertTrue(result['sources'])

    def test_lake_explanation_supported_without_invented_weather(self):
        with patch.object(self.app,'checkpoint',return_value=(None,{})):
            result=self.app.explain('modis','all','turbidity_proxy')
        self.assertEqual(result['provenance']['zone_id'],'all')
        self.assertFalse(result['forecast_summary']['available'])
        self.assertTrue(result['current_condition']['coverage_unknown'])
        self.assertEqual(result['hypotheses'][0]['status'],'Insufficient local support')
        self.assertIn('unavailable',result['hypotheses'][0]['support'][0])

    def test_training_validation_and_concurrency(self):
        for epochs in [0,301,True,'5']:
            with self.assertRaises(APIError): self.app.train({'dataset':'demo','epochs':epochs})
        with patch.object(self.app,'has_torch',return_value=True):
            self.app.training_lock.acquire()
            try:
                with self.assertRaises(APIError) as error:self.app.train({'epochs':1})
                self.assertEqual(error.exception.status,409)
            finally:self.app.training_lock.release()

    def test_async_train_saves_only_fingerprint_directory(self):
        app=Dashboard()
        completed=threading.Event()
        def mocked(df,output,epochs,progress_callback):
            self.assertIn('custom',str(output))
            self.assertEqual(epochs,1)
            progress_callback(dict(epoch=1,epochs=1,train_loss=.2,val_loss=.3,progress=80,message='Evaluating DET'))
            completed.set()
            return {'metrics':[]}
        with patch.object(app,'has_torch',return_value=True),patch('server.train_models',side_effect=mocked):
            job=app.train({'epochs':1})
            self.assertTrue(completed.wait(10))
            for thread in threading.enumerate():
                if thread.name=='aqualens-training':thread.join(10)
        self.assertEqual(app.job(job['id'])['status'],'complete')
        self.assertEqual(app.job(job['id'])['logs'][0]['epoch'],1)
        self.assertEqual(app.job(job['id'])['val_loss'],.3)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server=make_server(0)
        except PermissionError as exc:
            raise unittest.SkipTest(f'Loopback sockets blocked by sandbox: {exc}')
        try:
            probe=socket.create_connection(cls.server.server_address,timeout=2)
            probe.close()
        except PermissionError as exc:
            cls.server.server_close()
            raise unittest.SkipTest(f'Loopback connects blocked by sandbox: {exc}')
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.port=cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(3)

    def request(self,method,path,body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=60)
        conn.request(method,path,body=body,headers=headers or {})
        result=conn.getresponse()
        status,content,hdr=result.status,result.read(),dict(result.getheaders())
        conn.close()
        return status,content,hdr

    def test_bootstrap_and_export(self):
        status,body,headers=self.request('GET','/api/bootstrap')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['dataset'],'modis')
        self.assertEqual(headers['X-Content-Type-Options'],'nosniff')
        status,body,headers=self.request('GET','/api/export')
        self.assertEqual(status,200)
        self.assertIn(b'zone_id',body)
        self.assertIn('attachment',headers['Content-Disposition'])

    def test_foreign_origin_and_wrong_content_type(self):
        status,_,_=self.request('POST','/api/train','{}',{'Content-Type':'application/json','Origin':'https://evil.example'})
        self.assertEqual(status,403)
        status,_,_=self.request('POST','/api/upload','{}',{'Content-Type':'text/plain'})
        self.assertEqual(status,415)

    def test_errors_are_json_and_static_traversal_blocked(self):
        status,body,_=self.request('GET','/api/forecast?horizon_days=731')
        self.assertEqual(status,400)
        self.assertIn('error',json.loads(body))
        status,_,_=self.request('GET','/%2e%2e/server.py')
        self.assertEqual(status,404)
        status,_,_=self.request('GET','/api/bootstrap',headers={'Host':'evil.example'})
        self.assertEqual(status,403)

    def test_oversized_body_and_malformed_json(self):
        status,_,_=self.request('POST','/api/upload','{',{'Content-Type':'application/json'})
        self.assertEqual(status,400)
        status,_,_=self.request('POST','/api/upload',None,{'Content-Type':'application/json','Content-Length':str(MAX_BODY+1)})
        self.assertEqual(status,413)

if __name__=='__main__':unittest.main()


