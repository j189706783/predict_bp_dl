from pydantic import BaseModel

class pred_inputs(BaseModel):
    group:int
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
    loss:float
    scale:float
    pred1:float
    pred2:float
    upper:float
    lower:float