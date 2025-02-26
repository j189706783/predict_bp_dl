from pydantic import BaseModel

class pred_inputs(BaseModel):
    team:int
    wc:float
    gender:int
    rank:int
    old:float
    bwt:float
    sq:float
    ipf_gl_c:float 
    year:int

class pred_output(BaseModel):
    name:str
    pred:float
    upper:float
    lower:float