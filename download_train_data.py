from huggingface_hub import hf_hub_download                                                 
import shutil, os                        
                                                                                                
repo = "IffYuan/Embodied-R1-Dataset"
                                                                                            
files = [                                
    # Stage 1
    ("whatsup_rft_spatial_qa_train_4000_test_138_0417/train.parquet",
    "datasets/whatsup_rft_spatial_qa_train_4000_test_138_0417/train.parquet"),             
    ("whatsup_rft_spatial_qa_train_4000_test_138_0417/test.parquet",                        
    "datasets/whatsup_rft_spatial_qa_train_4000_test_138_0417/test.parquet"),              
    ("SAT_rft_spatial_qa_train_80000_test_4000_0428/train.parquet",                         
    "datasets/SAT_rft_spatial_qa_train_80000_test_4000_0428/train.parquet"),
    ("SAT_rft_spatial_qa_train_80000_test_4000_0428/test.parquet",                          
    "datasets/SAT_rft_spatial_qa_train_80000_test_4000_0428/test.parquet"),
    ("ViRL_rft_general_qa_train_17831_test_0_plus_0428/train.parquet",                      
    "datasets/ViRL_rft_general_qa_train_17831_test_0_plus_0428/train.parquet"),
                                                                                            
    # Stage 2                            
    ("robopoint_rft_point_ref_train_40000_test_2000_0417/train.parquet",                    
    "datasets/robopoint_rft_point_ref_train_40000_test_2000_0417/train.parquet"),
    ("FSD_points_rft_fsd_free_point_train_32790_test_300_0425/train.parquet",               
    "datasets/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/train.parquet"),
    ("FSD_points_rft_fsd_free_point_train_32790_test_300_0425/test.parquet",                
    "datasets/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/test.parquet"),      
    ("FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/train.parquet",                                                                                                 
    "datasets/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/train.parquet"),
    ("FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet",                                                                                           
    "datasets/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet"),   
    ("roborefit_rft_point_rec_train_35000_test_1000_0502/train.parquet",                    
    "datasets/roborefit_rft_point_rec_train_35000_test_1000_0502/train.parquet"),          
    ("roborefit_rft_point_rec_train_35000_test_1000_0502/test.parquet",                     
    "datasets/roborefit_rft_point_rec_train_35000_test_1000_0502/test.parquet"),
    ("refcoco_rft_point_rec_train_20000_test_189_0502/train.parquet",                       
    "datasets/refcoco_rft_point_rec_train_20000_test_189_0502/train.parquet"),
    ("refcoco_rft_point_rec_train_20000_test_189_0502/test.parquet",                        
    "datasets/refcoco_rft_point_rec_train_20000_test_189_0502/test.parquet"),
    ("handal_rft_grounding_rec_train_40000_test_1000_0503/train.parquet",                   
    "datasets/handal_rft_grounding_rec_train_40000_test_1000_0503/train.parquet"),
    ("handal_rft_grounding_rec_train_40000_test_1000_0503/test.parquet",                    
    "datasets/handal_rft_grounding_rec_train_40000_test_1000_0503/test.parquet"),
]                                                                                           
                                        
for remote, local in files:                                                                 
    os.makedirs(os.path.dirname(local), exist_ok=True)
    cached = hf_hub_download(repo, remote, repo_type="dataset")
    shutil.copy(cached, local)                                                              
    print(f"✓ {local}")