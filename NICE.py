def Breast_0(df):
    """
    Age < 30
    unexplained breast lump
    """
    Age = df.AGEDIAG
    Lump = df.Lump
    
    if (Age < 30) & Lump:
        return 1
    else:
        return 0
    
def Breast_1(df):
    """
    Age > 30
    unexplained breast lump
    or
    Age > 50
    discharge or retraction or other changes of concern
    """
    Age = df.AGEDIAG
    Lump = df.Lump
    Discharge = df.Discharge
    
    if (Age > 30) and Lump:
        return 1
    elif (Age > 50) and Discharge:
        return 1
    else:
        return 0
    
def Breast_2(df):
    """
    Skin changes that suggest breast cancer
    """
    Breast_Skin_Change = df.Breast_Skin_Change
    if Breast_Skin_Change:
        return 1
    else:
        return 0
    

def GI_Upper_1(df):
    """
    haematemesis
    ( “vomiting blood”)
    """
    haematemesis = 0
    if haematemesis:
        return 1
    else:
        return 0

def GI_Upper_2(df):
    """
    aged >= 55 and
    (dyspepsia or
    upper abdominal pain with low haemoglobin levels or
    raised platelet count with 
    ( nausea
    vomiting
    weight loss
    reflux
    dyspepsia
    upper abdominal pain) or
    nausea or vomiting with any of the following: 
    (weight loss
    reflux
    dyspepsia
    upper abdominal pain ))
    """
    Age = df.AGEDIAG
    Abdominal_Pain = df.Abdominal_Pain
    Anorexia = df.Anorexia
    
    Dyspepsia = df.unk_CC
    Upper_abdominal_pain = Abdominal_Pain
    Low_haemoglobin = df.unk_CC
    Raised_platelet_count = df.unk_CC
    N_V = df.N_V
    Weight_loss = Anorexia
    Reflux = df.unk_CC
    
    if (Age >= 55) and (Dyspepsia or Low_haemoglobin):
        return 1
    elif (Age >= 55) and Raised_platelet_count and (N_V or Weight_loss or Reflux or Dyspepsia or Upper_abdominal_pain):
        return 1
    elif (Age >= 55) and N_V and (Weight_loss or Reflux or Dyspepsia or Upper_abdominal_pain):
        return 1
    else:
        return 0
    
    
def GI_Liver_1(df):
    """
    upper abdominal mass consistent with an enlarged liver
    """
    
    return 0

def GI_Colorectal_1(df):
    """
    Age > 40 with unexplained weight loss and abdominal pain or
    Age > 50 with unexplained rectal bleeding or
    Age > 60 with (iron-deficiency anaemia or changes in their bowel habit)
    tests show occult blood in their faeces
    """
    Age = df.AGEDIAG
    Anorexia = df.Anorexia
    Abdominal_Pain = df.Abdominal_Pain
    Occult_Blood_Faeces = df.Occult_Blood_Faeces
    
    Dyspepsia = df.unk_CC
    Weight_loss = Anorexia
    Rectal_bleeding = Occult_Blood_Faeces
    Iron_deficiency_anaemia = df.unk_CC
    Changes_in_their_bowel_habit = df.unk_CC

    if (Age > 40) and Weight_loss and Abdominal_Pain:
        return 1    
    elif (Age > 50) and Rectal_bleeding :
        return 1
    elif (Age > 60) and (Iron_deficiency_anaemia or Changes_in_their_bowel_habit):
        return 1
    elif Occult_Blood_Faeces:
        return 1
    else:
        return 0
    
    
def GI_Colorectal_2(df):
    """
    Adult(>18) with a rectal or abdominal mass
    """
    Age = df.AGEDIAG
    
    return 0

def GI_Colorectal_3(df):
    """
    18 (adult) < Age < 50 with rectal bleeding and any of the following unexplained symptoms or findings:
    abdominal pain
    change in bowel habit 
    weight loss 
    iron-deficiency anaemia.
    """
    Age = df.AGEDIAG
    Anorexia = df.Anorexia
    Abdominal_Pain = df.Abdominal_Pain
    Occult_Blood_Faeces = df.Occult_Blood_Faeces
    
    Dyspepsia = df.unk_CC
    Weight_loss = Anorexia
    Occult_Blood_Faeces = df.Occult_Blood_Faeces
    
    Rectal_bleeding = Occult_Blood_Faeces
    Iron_deficiency_anaemia = df.unk_CC
    Changes_in_their_bowel_habit = df.unk_CC
    
    if (18 < Age < 50) and Weight_loss and (Abdominal_Pain or Changes_in_their_bowel_habit or Weight_loss or Iron_deficiency_anaemia):
        return 1
    else:
        return 0
    
def Gynae_Cervical_1(df):
    """
    Female if, on examination
    """
    Sex = df.isMale
    
    return 0

def Lung_1(df):
    """
    if they:
    have chest X-ray findings that suggest lung cancer or 
    are aged 40 and over with unexplained haemoptysis.
    """
    
    Age = df.AGEDIAG
    Hemoptysis = df.Hemoptysis
    
    if Age > 40 and Hemoptysis:
        return 1
    else:
        return 0

def Lung_2(df):
    """
    Age > 40 with 2 or more of (or 1 if they have ever smoked or Asbestos)
    cough
    fatigue
    shortness of breath 
    chest pain
    weight loss 
    appetite loss.
    """
    Age = df.AGEDIAG
    Cough = df.Cough
    Fatigue = df.Fatigue
    Dyspnea = df.Dyspnea
    Anorexia = df.Anorexia
    Chest_Pain = df.Chest_Pain
    
    
    Weight_loss = Anorexia
    Appetite_loss = Anorexia
    isSmoke = df.unk_CC
    isAsbestos = df.unk_CC

    if Age > 40 and ((Cough + Fatigue + Dyspnea + Chest_Pain + Weight_loss + Appetite_loss) >= 2):
        return 1
    elif Age > 40 and (isSmoke or isAsbestos) and ((Cough + Fatigue + Dyspnea + Chest_Pain + Weight_loss + Appetite_loss) >= 1):
        return 1
    else:
        return 0

def Lung_3(df):
    """
    Age > 40 with
    persistent or recurrent chest infection
    finger clubbing
    supraclavicular lymphadenopathy or persistent cervical lymphadenopathy chest signs consistent with lung cancer
    thrombocytosis.
    """
    Age = df.AGEDIAG
    Chronic_chest_infection = df.unk_CC
    Finger_clubbing = df.unk_CC
    Neck_lymphadenopathy = df.unk_CC
    Thrombocytosis = df.unk_CC
    
    if Age > 40 and (Chronic_chest_infection or Finger_clubbing or Neck_lymphadenopathy or Thrombocytosis):
        return 1
    else:
        return 0