# AD_Prediction_IGP
The following project is a practical application of the ML Model for AD Classification. The testing set structure goes as follows:

                  20 non-null     int64  
 1   Gender                     20 non-null     int64  
 2   Ethnicity                  20 non-null     int64  
 3   EducationLevel             20 non-null     int64  
 4   BMI                        20 non-null     float64
 5   Smoking                    20 non-null     int64  
 6   AlcoholConsumption         20 non-null     float64
 7   PhysicalActivity           20 non-null     float64
 8   DietQuality                20 non-null     float64
 9   SleepQuality               20 non-null     float64
 10  FamilyHistoryAlzheimers    20 non-null     int64  
 11  CardiovascularDisease      20 non-null     int64  
 12  Diabetes                   20 non-null     int64  
 13  Depression                 20 non-null     int64  
 14  HeadInjury                 20 non-null     int64  
 15  Hypertension               20 non-null     int64  
 16  SystolicBP                 20 non-null     int64  
 17  DiastolicBP                20 non-null     int64  
 18  CholesterolTotal           20 non-null     float64
 19  CholesterolLDL             20 non-null     float64
 20  CholesterolHDL             20 non-null     float64
 21  CholesterolTriglycerides   20 non-null     float64
 22  MMSE                       20 non-null     float64
 23  FunctionalAssessment       20 non-null     float64
 24  MemoryComplaints           20 non-null     int64  
 25  BehavioralProblems         20 non-null     int64  
 26  ADL                        20 non-null     float64
 27  Confusion                  20 non-null     int64  
 28  Disorientation             20 non-null     int64  
 29  PersonalityChanges         20 non-null     int64  
 30  DifficultyCompletingTasks  20 non-null     int64  
 31  Forgetfulness              20 non-null     int64  


 Sample data in Json format:
 {"Age":{"6293":78},"Gender":{"6293":0},"Ethnicity":{"6293":0},"EducationLevel":{"6293":1},"BMI":{"6293":39.7849654908},"Smoking":{"6293":0},"AlcoholConsumption":{"6293":3.2121950272},"PhysicalActivity":{"6293":2.4138940078},"DietQuality":{"6293":5.1498566021},"SleepQuality":{"6293":5.9448905176},"FamilyHistoryAlzheimers":{"6293":1},"CardiovascularDisease":{"6293":0},"Diabetes":{"6293":0},"Depression":{"6293":0},"HeadInjury":{"6293":1},"Hypertension":{"6293":0},"SystolicBP":{"6293":157},"DiastolicBP":{"6293":76},"CholesterolTotal":{"6293":225.027759402},"CholesterolLDL":{"6293":157.4768503313},"CholesterolHDL":{"6293":81.0871018552},"CholesterolTriglycerides":{"6293":307.5640224577},"MMSE":{"6293":9.5767862856},"FunctionalAssessment":{"6293":1.3101354173},"MemoryComplaints":{"6293":0},"BehavioralProblems":{"6293":0},"ADL":{"6293":5.1729572534},"Confusion":{"6293":0},"Disorientation":{"6293":0},"PersonalityChanges":{"6293":0},"DifficultyCompletingTasks":{"6293":0},"Forgetfulness":{"6293":0}}
 

 
