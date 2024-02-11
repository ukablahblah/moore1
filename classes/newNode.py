class newNode:
    name_to_io = {'': (0,0),
         'A2O1A1O1Ixp25_ASAP7_75t_L':(5,1),
         'AND2x2_ASAP7_75t_L':(2,1),
         'AND2x6_ASAP7_75t_L':(2,1),
         'AND3x1_ASAP7_75t_L':(3,1),
         'AND3x4_ASAP7_75t_L':(3,1),
         'AND4x2_ASAP7_75t_L':(4,1),
         'AO211x2_ASAP7_75t_L':(4,1),
         'AO21x2_ASAP7_75t_L':(3,1),
         'AO221x2_ASAP7_75t_L':(5,1),
         'AO222x2_ASAP7_75t_L':(6,1),
         'AO22x2_ASAP7_75t_L':(4,1),
         'AO31x2_ASAP7_75t_L':(4,1),
         'AO32x2_ASAP7_75t_L':(5,1),
         'AO33x2_ASAP7_75t_L':(6,1),
         'AOI211xp5_ASAP7_75t_L':(4,1),
         'AOI21xp5_ASAP7_75t_L':(4,1),
         'AOI221xp5_ASAP7_75t_L':(5,1),
         'AOI222xp33_ASAP7_75t_L':(6,1),
         'AOI22xp5_ASAP7_75t_L':(4,1),
         'AOI31xp67_ASAP7_75t_L':(4,1),
         'AOI32xp33_ASAP7_75t_L':(5,1),
         'AOI33xp33_ASAP7_75t_L':(6,1),
         'BUFx8_ASAP7_75t_SL':(1,1),
         'DFFLQx4_ASAP7_75t_L':(3,1),
         'FAx1_ASAP7_75t_L':(4,1),
         'INVxp67_ASAP7_75t_SL':(1,1),
         'NAND2xp5_ASAP7_75t_L':(2,1),
         'NAND2xp67_ASAP7_75t_L':(2,1),
         'NAND3xp33_ASAP7_75t_L':(3,1),
         'NAND4xp75_ASAP7_75t_L':(4,1),
         'NOR2xp67_ASAP7_75t_L':(2,1),
         'NOR3xp33_ASAP7_75t_L':(3,1),
         'NOR4xp75_ASAP7_75t_L':(4,1),
         'OA211x2_ASAP7_75t_L':(4,1),
         'OA21x2_ASAP7_75t_L':(3,1),
         'OA221x2_ASAP7_75t_L':(5,1),
         'OA222x2_ASAP7_75t_L':(6,1),
         'OA22x2_ASAP7_75t_L':(4,1),
         'OA31x2_ASAP7_75t_L':(4,1),
         'OA33x2_ASAP7_75t_L':(6,1),
         'OAI211xp5_ASAP7_75t_L':(4,1),
         'OAI21xp5_ASAP7_75t_L':(3,1),
         'OAI221xp5_ASAP7_75t_L':(5,1),
         'OAI222xp33_ASAP7_75t_L':(6,1),
         'OAI22xp5_ASAP7_75t_L':(4,1),
         'OAI31xp67_ASAP7_75t_L':(4,1),
         'OAI32xp33_ASAP7_75t_L':(5,1),
         'OAI33xp33_ASAP7_75t_L':(6,1),
         'OR2x6_ASAP7_75t_L':(2,1),
         'OR3x4_ASAP7_75t_L':(3,1),
         'OR4x2_ASAP7_75t_L':(4,1),
         'XNOR2xp5_ASAP7_75t_L':(2,1),
         'XOR2xp5_ASAP7_75t_L':(2,1)}
    
    def __init__(self,inputs:int, outputs:int, input_val:int, output_val:int, cell_name:str, signal_type:str):
        self.inputs = inputs
        self.outputs = outputs
        self.output_vals = output_val
        self.input_vals = input_val
        self.cell_name = cell_name
        self.signal_type = signal_type

    def __init__(self,old):
        self.name = old.name
        self.output_vals = None
        self.input_vals = None
        self.cell_name = old.cell
        self.signal_type = None
        self.inputs = self.name_to_io[self.cell_name][0]
        self.outputs = self.name_to_io[self.cell_name][1]