import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import keras_cv
import keras
import gzip
import html
from functools import lru_cache
import regex as re

_UNCONDITIONAL_TOKENS = [
    49406,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
    49407,
]
_ALPHAS_CUMPROD = [
    0.99915,
    0.998296,
    0.9974381,
    0.9965762,
    0.99571025,
    0.9948404,
    0.9939665,
    0.9930887,
    0.9922069,
    0.9913211,
    0.9904313,
    0.98953754,
    0.9886398,
    0.9877381,
    0.9868324,
    0.98592263,
    0.98500896,
    0.9840913,
    0.9831696,
    0.982244,
    0.98131436,
    0.9803808,
    0.97944313,
    0.97850156,
    0.977556,
    0.9766064,
    0.97565293,
    0.9746954,
    0.9737339,
    0.9727684,
    0.97179896,
    0.97082555,
    0.96984816,
    0.96886677,
    0.9678814,
    0.96689206,
    0.96589875,
    0.9649015,
    0.96390027,
    0.9628951,
    0.9618859,
    0.96087277,
    0.95985574,
    0.95883465,
    0.9578097,
    0.95678073,
    0.95574784,
    0.954711,
    0.95367026,
    0.9526256,
    0.9515769,
    0.95052433,
    0.94946784,
    0.94840735,
    0.947343,
    0.94627476,
    0.9452025,
    0.9441264,
    0.9430464,
    0.9419625,
    0.9408747,
    0.939783,
    0.9386874,
    0.93758786,
    0.9364845,
    0.93537724,
    0.9342661,
    0.9331511,
    0.9320323,
    0.9309096,
    0.929783,
    0.9286526,
    0.9275183,
    0.9263802,
    0.92523825,
    0.92409253,
    0.92294294,
    0.9217895,
    0.92063236,
    0.9194713,
    0.9183065,
    0.9171379,
    0.91596556,
    0.9147894,
    0.9136095,
    0.91242576,
    0.9112383,
    0.9100471,
    0.9088522,
    0.9076535,
    0.9064511,
    0.90524495,
    0.9040351,
    0.90282154,
    0.9016043,
    0.90038335,
    0.8991587,
    0.8979304,
    0.8966984,
    0.89546275,
    0.89422345,
    0.8929805,
    0.89173394,
    0.89048374,
    0.88922995,
    0.8879725,
    0.8867115,
    0.88544685,
    0.88417864,
    0.88290685,
    0.8816315,
    0.88035256,
    0.8790701,
    0.87778413,
    0.8764946,
    0.8752016,
    0.873905,
    0.87260497,
    0.8713014,
    0.8699944,
    0.86868393,
    0.86737,
    0.8660526,
    0.8647318,
    0.86340755,
    0.8620799,
    0.8607488,
    0.85941434,
    0.8580765,
    0.8567353,
    0.8553907,
    0.8540428,
    0.85269153,
    0.85133696,
    0.84997904,
    0.84861785,
    0.8472533,
    0.8458856,
    0.8445145,
    0.84314024,
    0.84176266,
    0.8403819,
    0.8389979,
    0.8376107,
    0.8362203,
    0.83482677,
    0.83343,
    0.8320301,
    0.8306271,
    0.8292209,
    0.82781166,
    0.82639927,
    0.8249838,
    0.82356524,
    0.8221436,
    0.82071894,
    0.81929123,
    0.81786054,
    0.8164268,
    0.8149901,
    0.8135504,
    0.81210774,
    0.81066215,
    0.8092136,
    0.8077621,
    0.80630773,
    0.80485046,
    0.8033903,
    0.80192727,
    0.8004614,
    0.79899275,
    0.79752123,
    0.7960469,
    0.7945698,
    0.7930899,
    0.79160726,
    0.7901219,
    0.7886338,
    0.787143,
    0.7856495,
    0.7841533,
    0.78265446,
    0.78115296,
    0.7796488,
    0.77814204,
    0.7766327,
    0.7751208,
    0.7736063,
    0.77208924,
    0.7705697,
    0.7690476,
    0.767523,
    0.7659959,
    0.7644664,
    0.76293445,
    0.7614,
    0.7598632,
    0.75832397,
    0.75678235,
    0.75523835,
    0.75369203,
    0.7521434,
    0.75059247,
    0.7490392,
    0.7474837,
    0.7459259,
    0.7443659,
    0.74280363,
    0.7412392,
    0.7396726,
    0.7381038,
    0.73653287,
    0.7349598,
    0.7333846,
    0.73180735,
    0.730228,
    0.7286466,
    0.7270631,
    0.7254777,
    0.72389024,
    0.72230077,
    0.7207094,
    0.71911603,
    0.7175208,
    0.7159236,
    0.71432453,
    0.7127236,
    0.71112084,
    0.7095162,
    0.7079098,
    0.7063016,
    0.70469165,
    0.70307994,
    0.7014665,
    0.69985133,
    0.6982345,
    0.696616,
    0.6949958,
    0.69337404,
    0.69175065,
    0.69012564,
    0.6884991,
    0.68687093,
    0.6852413,
    0.68361014,
    0.6819775,
    0.6803434,
    0.67870784,
    0.6770708,
    0.6754324,
    0.6737926,
    0.67215145,
    0.670509,
    0.66886514,
    0.66722,
    0.6655736,
    0.66392595,
    0.662277,
    0.6606269,
    0.65897554,
    0.657323,
    0.65566933,
    0.6540145,
    0.6523586,
    0.6507016,
    0.6490435,
    0.64738435,
    0.6457241,
    0.64406294,
    0.6424008,
    0.64073765,
    0.63907355,
    0.63740855,
    0.6357426,
    0.6340758,
    0.6324082,
    0.6307397,
    0.6290704,
    0.6274003,
    0.6257294,
    0.62405777,
    0.6223854,
    0.62071234,
    0.6190386,
    0.61736417,
    0.6156891,
    0.61401343,
    0.6123372,
    0.6106603,
    0.6089829,
    0.607305,
    0.6056265,
    0.6039476,
    0.60226816,
    0.6005883,
    0.598908,
    0.59722733,
    0.5955463,
    0.59386486,
    0.5921831,
    0.59050107,
    0.5888187,
    0.5871361,
    0.5854532,
    0.5837701,
    0.5820868,
    0.5804033,
    0.5787197,
    0.5770359,
    0.575352,
    0.57366806,
    0.571984,
    0.5702999,
    0.5686158,
    0.56693166,
    0.56524754,
    0.5635635,
    0.5618795,
    0.56019557,
    0.5585118,
    0.5568281,
    0.55514455,
    0.5534612,
    0.551778,
    0.5500951,
    0.5484124,
    0.54673,
    0.5450478,
    0.54336596,
    0.54168445,
    0.54000324,
    0.53832245,
    0.5366421,
    0.53496206,
    0.5332825,
    0.53160346,
    0.5299248,
    0.52824676,
    0.5265692,
    0.52489215,
    0.5232157,
    0.5215398,
    0.51986456,
    0.51818997,
    0.51651603,
    0.51484275,
    0.5131702,
    0.5114983,
    0.5098272,
    0.50815684,
    0.5064873,
    0.50481856,
    0.50315064,
    0.50148356,
    0.4998174,
    0.4981521,
    0.49648774,
    0.49482432,
    0.49316183,
    0.49150035,
    0.48983985,
    0.4881804,
    0.486522,
    0.48486462,
    0.4832084,
    0.48155323,
    0.4798992,
    0.47824633,
    0.47659463,
    0.4749441,
    0.47329482,
    0.4716468,
    0.47,
    0.46835446,
    0.46671024,
    0.46506736,
    0.4634258,
    0.46178558,
    0.46014675,
    0.45850933,
    0.45687333,
    0.45523876,
    0.45360568,
    0.45197406,
    0.45034397,
    0.44871536,
    0.44708833,
    0.44546285,
    0.44383895,
    0.44221666,
    0.440596,
    0.43897697,
    0.43735963,
    0.43574396,
    0.43412998,
    0.43251774,
    0.43090722,
    0.4292985,
    0.42769152,
    0.42608637,
    0.42448303,
    0.4228815,
    0.42128187,
    0.4196841,
    0.41808826,
    0.4164943,
    0.4149023,
    0.41331223,
    0.41172415,
    0.41013804,
    0.40855396,
    0.4069719,
    0.4053919,
    0.40381396,
    0.4022381,
    0.40066436,
    0.39909273,
    0.39752322,
    0.3959559,
    0.39439073,
    0.39282778,
    0.39126703,
    0.3897085,
    0.3881522,
    0.3865982,
    0.38504648,
    0.38349706,
    0.38194993,
    0.38040516,
    0.37886274,
    0.37732267,
    0.375785,
    0.37424973,
    0.37271687,
    0.37118647,
    0.36965853,
    0.36813304,
    0.36661002,
    0.36508954,
    0.36357155,
    0.3620561,
    0.36054322,
    0.3590329,
    0.35752517,
    0.35602003,
    0.35451752,
    0.35301763,
    0.3515204,
    0.3500258,
    0.3485339,
    0.3470447,
    0.34555823,
    0.34407446,
    0.34259343,
    0.34111515,
    0.33963963,
    0.33816692,
    0.336697,
    0.3352299,
    0.33376563,
    0.3323042,
    0.33084565,
    0.32938993,
    0.32793713,
    0.3264872,
    0.32504022,
    0.32359615,
    0.32215503,
    0.32071686,
    0.31928164,
    0.31784943,
    0.3164202,
    0.314994,
    0.3135708,
    0.31215066,
    0.31073356,
    0.3093195,
    0.30790854,
    0.30650064,
    0.30509588,
    0.30369422,
    0.30229566,
    0.30090025,
    0.299508,
    0.2981189,
    0.29673296,
    0.29535022,
    0.2939707,
    0.29259437,
    0.29122123,
    0.28985137,
    0.28848472,
    0.28712133,
    0.2857612,
    0.28440437,
    0.2830508,
    0.28170055,
    0.2803536,
    0.27900997,
    0.27766964,
    0.27633268,
    0.27499905,
    0.2736688,
    0.27234194,
    0.27101842,
    0.2696983,
    0.26838157,
    0.26706827,
    0.26575837,
    0.26445192,
    0.26314887,
    0.2618493,
    0.26055318,
    0.2592605,
    0.25797132,
    0.2566856,
    0.2554034,
    0.25412467,
    0.25284946,
    0.25157773,
    0.2503096,
    0.24904492,
    0.24778382,
    0.24652626,
    0.24527225,
    0.2440218,
    0.24277493,
    0.24153163,
    0.24029191,
    0.23905578,
    0.23782326,
    0.23659433,
    0.23536903,
    0.23414734,
    0.23292927,
    0.23171483,
    0.23050404,
    0.22929688,
    0.22809339,
    0.22689353,
    0.22569734,
    0.22450483,
    0.22331597,
    0.2221308,
    0.22094932,
    0.21977153,
    0.21859743,
    0.21742703,
    0.21626033,
    0.21509734,
    0.21393807,
    0.21278252,
    0.21163069,
    0.21048258,
    0.20933822,
    0.20819758,
    0.2070607,
    0.20592754,
    0.20479813,
    0.20367248,
    0.20255059,
    0.20143245,
    0.20031808,
    0.19920748,
    0.19810064,
    0.19699757,
    0.19589828,
    0.19480278,
    0.19371104,
    0.1926231,
    0.19153893,
    0.19045855,
    0.18938197,
    0.18830918,
    0.18724018,
    0.18617497,
    0.18511358,
    0.18405597,
    0.18300217,
    0.18195218,
    0.18090598,
    0.1798636,
    0.17882504,
    0.17779027,
    0.1767593,
    0.17573217,
    0.17470883,
    0.1736893,
    0.1726736,
    0.1716617,
    0.17065361,
    0.16964935,
    0.1686489,
    0.16765225,
    0.16665943,
    0.16567042,
    0.16468522,
    0.16370384,
    0.16272627,
    0.16175252,
    0.16078258,
    0.15981644,
    0.15885411,
    0.1578956,
    0.15694089,
    0.15599,
    0.15504292,
    0.15409963,
    0.15316014,
    0.15222447,
    0.15129258,
    0.1503645,
    0.14944021,
    0.14851972,
    0.14760303,
    0.14669013,
    0.14578101,
    0.14487568,
    0.14397413,
    0.14307636,
    0.14218238,
    0.14129217,
    0.14040573,
    0.13952307,
    0.13864417,
    0.13776903,
    0.13689767,
    0.13603005,
    0.13516618,
    0.13430607,
    0.13344972,
    0.1325971,
    0.13174823,
    0.1309031,
    0.13006169,
    0.12922402,
    0.12839006,
    0.12755983,
    0.12673332,
    0.12591052,
    0.12509143,
    0.12427604,
    0.12346435,
    0.12265636,
    0.121852055,
    0.12105144,
    0.1202545,
    0.11946124,
    0.11867165,
    0.11788572,
    0.11710346,
    0.11632485,
    0.115549885,
    0.11477857,
    0.11401089,
    0.11324684,
    0.11248643,
    0.11172963,
    0.11097645,
    0.110226884,
    0.10948092,
    0.10873855,
    0.10799977,
    0.107264586,
    0.106532976,
    0.105804935,
    0.10508047,
    0.10435956,
    0.1036422,
    0.10292839,
    0.10221813,
    0.1015114,
    0.10080819,
    0.100108504,
    0.09941233,
    0.098719664,
    0.0980305,
    0.09734483,
    0.09666264,
    0.09598393,
    0.095308684,
    0.09463691,
    0.093968585,
    0.09330372,
    0.092642285,
    0.09198428,
    0.09132971,
    0.09067855,
    0.090030804,
    0.089386456,
    0.088745505,
    0.088107936,
    0.08747375,
    0.08684293,
    0.08621547,
    0.085591376,
    0.084970616,
    0.08435319,
    0.0837391,
    0.08312833,
    0.08252087,
    0.08191671,
    0.08131585,
    0.08071827,
    0.080123976,
    0.07953294,
    0.078945175,
    0.078360654,
    0.077779375,
    0.07720133,
    0.07662651,
    0.07605491,
    0.07548651,
    0.07492131,
    0.0743593,
    0.07380046,
    0.073244795,
    0.07269229,
    0.07214294,
    0.07159673,
    0.07105365,
    0.070513695,
    0.06997685,
    0.069443114,
    0.06891247,
    0.06838491,
    0.067860425,
    0.06733901,
    0.066820644,
    0.06630533,
    0.06579305,
    0.0652838,
    0.06477757,
    0.06427433,
    0.0637741,
    0.063276865,
    0.06278259,
    0.062291294,
    0.061802953,
    0.06131756,
    0.0608351,
    0.060355574,
    0.05987896,
    0.059405252,
    0.058934443,
    0.05846652,
    0.058001474,
    0.057539295,
    0.05707997,
    0.056623492,
    0.05616985,
    0.05571903,
    0.055271026,
    0.054825824,
    0.05438342,
    0.053943794,
    0.053506944,
    0.05307286,
    0.052641522,
    0.052212927,
    0.051787063,
    0.051363923,
    0.05094349,
    0.050525755,
    0.05011071,
    0.04969834,
    0.049288645,
    0.0488816,
    0.048477206,
    0.048075445,
    0.04767631,
    0.047279786,
    0.04688587,
    0.046494544,
    0.046105802,
    0.04571963,
    0.04533602,
    0.04495496,
    0.04457644,
    0.044200446,
    0.04382697,
    0.043456003,
    0.043087535,
    0.042721547,
    0.042358037,
    0.04199699,
    0.041638397,
    0.041282244,
    0.040928524,
    0.040577225,
    0.040228333,
    0.039881844,
    0.039537743,
    0.039196018,
    0.038856663,
    0.038519662,
    0.038185004,
    0.037852682,
    0.037522685,
    0.037195,
    0.036869615,
    0.036546525,
    0.036225714,
    0.03590717,
    0.035590887,
    0.035276853,
    0.034965057,
    0.034655485,
    0.03434813,
    0.03404298,
    0.033740025,
    0.033439253,
    0.033140652,
    0.032844216,
    0.03254993,
    0.032257784,
    0.03196777,
    0.031679876,
    0.031394087,
    0.031110398,
    0.030828796,
    0.030549273,
    0.030271813,
    0.02999641,
    0.029723052,
    0.029451728,
    0.029182427,
    0.02891514,
    0.028649855,
    0.028386563,
    0.028125253,
    0.02786591,
    0.027608532,
    0.027353102,
    0.027099613,
    0.026848052,
    0.026598409,
    0.026350675,
    0.02610484,
    0.02586089,
    0.02561882,
    0.025378617,
    0.025140269,
    0.024903767,
    0.0246691,
    0.02443626,
    0.024205236,
    0.023976017,
    0.023748592,
    0.023522953,
    0.023299087,
    0.023076987,
    0.022856642,
    0.02263804,
    0.022421172,
    0.022206029,
    0.0219926,
    0.021780876,
    0.021570845,
    0.021362498,
    0.021155827,
    0.020950818,
    0.020747466,
    0.020545758,
    0.020345684,
    0.020147236,
    0.019950403,
    0.019755175,
    0.019561544,
    0.019369498,
    0.019179028,
    0.018990126,
    0.01880278,
    0.018616982,
    0.018432721,
    0.01824999,
    0.018068777,
    0.017889075,
    0.017710872,
    0.01753416,
    0.017358929,
    0.017185168,
    0.017012872,
    0.016842028,
    0.016672628,
    0.016504662,
    0.016338123,
    0.016173,
    0.016009282,
    0.015846964,
    0.015686033,
    0.015526483,
    0.015368304,
    0.015211486,
    0.0150560215,
    0.014901901,
    0.014749114,
    0.014597654,
    0.014447511,
    0.0142986765,
    0.014151142,
    0.014004898,
    0.013859936,
    0.013716248,
    0.0135738235,
    0.013432656,
    0.013292736,
    0.013154055,
    0.013016605,
    0.012880377,
    0.012745362,
    0.012611552,
    0.012478939,
    0.012347515,
    0.01221727,
    0.012088198,
    0.0119602885,
    0.0118335355,
    0.011707929,
    0.011583461,
    0.011460125,
    0.011337912,
    0.011216813,
    0.011096821,
    0.010977928,
    0.0108601255,
    0.010743406,
    0.010627762,
    0.0105131855,
    0.010399668,
    0.010287202,
    0.01017578,
    0.010065395,
    0.009956039,
    0.009847702,
    0.009740381,
    0.0096340645,
    0.009528747,
    0.009424419,
    0.009321076,
    0.009218709,
    0.00911731,
    0.009016872,
    0.008917389,
    0.008818853,
    0.008721256,
    0.008624591,
    0.008528852,
    0.00843403,
    0.00834012,
    0.008247114,
    0.008155004,
    0.008063785,
    0.007973449,
    0.007883989,
    0.007795398,
    0.0077076694,
    0.0076207966,
    0.0075347726,
    0.007449591,
    0.0073652444,
    0.007281727,
    0.0071990318,
    0.007117152,
    0.0070360815,
    0.0069558136,
    0.0068763415,
    0.006797659,
    0.00671976,
    0.0066426382,
    0.0065662866,
    0.006490699,
    0.0064158696,
    0.006341792,
    0.00626846,
    0.0061958674,
    0.0061240084,
    0.0060528764,
    0.0059824656,
    0.0059127696,
    0.0058437833,
    0.0057755,
    0.0057079145,
    0.00564102,
    0.0055748112,
    0.0055092825,
    0.005444428,
    0.005380241,
    0.0053167176,
    0.005253851,
    0.005191636,
    0.005130066,
    0.0050691366,
    0.0050088423,
    0.0049491767,
    0.004890135,
    0.0048317118,
    0.004773902,
    0.004716699,
    0.0046600983,
]
class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)
    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)
class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        # 逐点卷积,切换通道
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)
    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw_q, c
        k = ops.transpose(k, (0, 3, 1, 2)) # (b,c,h,w)
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw_k
        y = q @ k # (b,hw_q,hw_k)
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype)) # scale
        # softmax,在k的空间序列上归一化,这样就得到q对k查询的注意力权重
        # 权重值大的表示与q高度相关,权重小的表示与q相对无关
        y = keras.activations.softmax(y) # (b,hw_q,hw_k) 
        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2)) # (b,c,h,w)
        v = ops.reshape(v, (-1, c, h * w)) # (b,c,hw_v)
        y = ops.transpose(y, (0, 2, 1)) # (b,hw_k,hw_q)
        x = v @ y # (b,c,hw_v)@(b,hw_k,hw_q)-->(b,c,hw_q)
        x = ops.transpose(x, (0, 2, 1)) # (b,hw_q,c)
        x = ops.reshape(x, (-1, h, w, c)) # (b,h_q,w_q,c)
        return self.proj_out(x) + inputs # 注意力前后残差

# 建立字节到字符的映射
@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # bs是一个包含字节值的列表(0--255)
    # cs中既包括了原始索引对应的字符,也包括了一些
    # 追加的,这些追加的字符替换了原先索引位置对应的字符
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    # 定义的从字节到字符的映射
    return dict(zip(bs, cs))
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    # 通过不断改变prev_char指向的值,来获取所有的字符对
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
# <字符在HTML中被转义为&lt;，>被转义为&gt;，&本身被转义为&amp;。
# html.unescape()函数接受一个字符串作为输入，该字符串可能包含HTML转义字符，并返回一个新的字符串，
# 其中所有的HTML转义字符都被转换回了它们所代表的原始字符,就是把&amp;转换回&
def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()
# 使用 re.sub(r"\s+", " ", text) 将文本中的所有连续空白字符替换为一个空格。这里的正则
# 表达式 r"\s+" 匹配一个或多个（由 + 指定）空白字符，并将它们替换为单个空格 " "。注意，r
# 前缀表示这是一个原始字符串，意味着在这个字符串中的反斜杠 \ 将不会被当作转义字符处理，这
# 对于正则表达式来说是必要的，因为正则表达式中经常需要使用到反斜杠。
# 使用 text.strip() 去除字符串开头和结尾的所有空白字符（包括空格、换行符、制表符等）。
# strip() 方法不会修改原字符串，而是返回一个新的字符串。
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def td_dot(a, b):
    aa = ops.reshape(a, (-1, a.shape[2], a.shape[3])) # (bh,s_q,dk)
    bb = ops.reshape(b, (-1, b.shape[2], b.shape[3])) # (bh,dk,s_k)
    cc = keras.layers.Dot(axes=(2, 1))([aa, bb]) # (bh,s_q,s_k)
    # (b,h,s_q,s_k)
    return ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))

class CrossAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(num_heads * head_size) # d_model

    def call(self, inputs, context=None):
        if context is None:
            context = inputs
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = ops.reshape( # (b,s,h,dk)
            q, (-1, inputs.shape[1], self.num_heads, self.head_size)
        )
        k = ops.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = ops.reshape( # (b,s,h,dk)
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = ops.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = ops.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = ops.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale # (b,h,s_q,s_k)
        weights = keras.activations.softmax( # 在s_k上归一化
            score
        )  
        # (b,h,s_q,s_k)@(b,h,s_v,dk)-->(b,h,s_q,dk)
        attn = td_dot(weights, v)
        attn = ops.transpose( # (b,s_q,h,dk)
            attn, (0, 2, 1, 3)
        )  # (b,s,d_model)
        out = ops.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)

class CLIPAttention(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads # dk
        self.scale = self.head_dim**-0.5 
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    def reshape_states(self, x, sequence_length, batch_size):
        # (b,s,h,dk)
        x = ops.reshape(
            x, (batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        return ops.transpose( # (b,h,s,dk)
            x, (0, 2, 1, 3)
        ) 
    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = ops.shape(inputs)[1] # s
            attention_mask = ops.triu( # 因果掩码
                ops.ones((1, 1, length, length), dtype=self.compute_dtype)
                * -float("inf"),
                k=1,
            )
        _, tgt_len, embed_dim = inputs.shape # s,d
        query_states = self.q_proj(inputs) * self.scale # (b,s,d)
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1) # (b,h,s,dk) 
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1) # (b,s,h,dk)
        proj_shape = (-1, tgt_len, self.head_dim) # (bh,s,dk)
        query_states = self.reshape_states(query_states, tgt_len, -1) # (b,s,h,dk)
        query_states = ops.reshape(query_states, proj_shape) # (bh,s,dk)
        key_states = ops.reshape(key_states, proj_shape) # (bh,s,dk)

        src_len = tgt_len
        value_states = ops.reshape(value_states, proj_shape) # (bh,s,dk)
        # (bh,s_q,dk)@(bh,dk,s_k)-->(bh,s_q,s_k)
        attn_weights = query_states @ ops.transpose(key_states, (0, 2, 1))
        # (b,h,s_q,s_k)
        attn_weights = ops.reshape(
            attn_weights, (-1, self.num_heads, tgt_len, src_len)
        )
        attn_weights = attn_weights + attention_mask # 遮挡未来token
        attn_weights = ops.reshape(attn_weights, (-1, tgt_len, src_len)) #(bh,s_q,s_k)
        # 在s_k上求softmax,这样就获取到编码器输出的注意力权重
        attn_weights = ops.softmax(attn_weights, axis=-1)
        # (bh,s_q,s_k)@(bh,s_v,dk)-->(bh,s_q,dk) 
        # 获取query的注意力token
        attn_output = attn_weights @ value_states
        # (b,h,s_q,dk)
        attn_output = ops.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3)) # (b,s_q,h,dk)
        attn_output = ops.reshape(attn_output, (-1, tgt_len, embed_dim))
        return self.out_proj(attn_output)

class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2) # 上采样
        self.conv = PaddedConv2D(channels, 3, padding=1)
    def call(self, inputs):
        return self.conv(self.ups(inputs))

class CLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = keras.layers.Dense(embed_dim * 4)
        self.fc2 = keras.layers.Dense(embed_dim)
        self.activation = activation
    def call(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x  # 自注意力前后残差
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual # 前馈前后残差

class CLIPEmbedding(keras.layers.Layer):
    def __init__(
        self, input_dim=49408, output_dim=768, max_length=77, **kwargs
    ):
        super().__init__(**kwargs)
        self.token_embedding = keras.layers.Embedding(input_dim, output_dim)
        self.position_embedding = keras.layers.Embedding(max_length, output_dim)

    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        return tokens + positions

def quick_gelu(x):
    return x * ops.sigmoid(x * 1.702) 

class TextEncoder(keras.Model):
    def __init__(
        self, max_length, vocab_size=49408, name=None, download_weights=True
    ):
        tokens = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="tokens"
        )
        positions = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="positions"
        )
        # token和位置嵌入,一起作为某个位置token的表示
        x = CLIPEmbedding(vocab_size,512, max_length)([tokens, positions])
        # transformer encoder
        for _ in range(6):
            x = CLIPEncoderLayer(512, 12, activation=quick_gelu)(x)
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([tokens, positions], embedded, name=name)

class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)
class ResnetBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)
class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # 这是因为 GEGLU 层将输出分为两部分：一部分是线性变换的结果（x），另一部分是用于门控机制
        # 的结果（gate）。
        self.dense = keras.layers.Dense(output_dim * 2)
    # 首先通过 self.dense 层对输入进行变换，得到两倍于 output_dim 的输出。
    # 然后，将输出 x 分割成两部分：x 本身（线性部分）和 gate（门控部分）。这两部分的维度都是 
    # output_dim。
    def call(self, inputs):
        # 当Dense层接收到输入向量时，它会根据训练过程中学习到的权重矩阵和偏置项来计算输出。这些权重和偏置是通过反向传播
        # 算法在训练过程中自动调整的，以最小化某个损失函数。因此，输出向量中的每个值都是输入向量中值的线性组合（加上偏置
        # 项），而不是简单的复制。
        x = self.dense(inputs)
        # 接下来，对 gate 应用一个特定的非线性变换。这个变换首先乘以一个缩放因子 0.7978845608（可能是为
        # 了数值稳定性或缩放输出范围），然后加上一个基于 gate 平方的项 0.044715 * (gate**2)。这个非线性
        # 变换可能是为了引入更多的非线性能力，但具体选择这个特定形式的原因可能依赖于实验或理论背景。
        # 使用 tanh 激活函数对非线性变换后的结果进行激活，得到 tanh_res。
        # 最后，根据 GEGLU 的定义，输出是 x 乘以 0.5 * gate * (1 + tanh_res)。这里，gate 和其非线性变换后的
        # 版本共同决定了 x 的哪些部分应该被保留或抑制，从而实现了门控机制。,切片操作分成两部分
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)
# 普通transformer decoder
class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4) # 这里充当前馈层
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        inputs, context = inputs
        # 自注意力
        x = self.attn1(self.norm1(inputs), context=None) + inputs
        # 跨注意力
        x = self.attn2(self.norm2(x), context=context) + x
        # 前馈前后残差
        return self.dense(self.geglu(self.norm3(x))) + x

class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size # h*dk-->d
        if fully_connected: # 文本数据
            self.proj1 = keras.layers.Dense(num_heads * head_size)
        else: # 图片数据
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        # 基本的transfomer
        self.transformer_block = BasicTransformerBlock(
            channels, num_heads, head_size
        )
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x) # 切换通道
        x = ops.reshape(x, (-1, h * w, c)) # (b,hw,c)
        x = self.transformer_block([x, context]) # (n,s,c)
        x = ops.reshape(x, (-1, h, w, c)) # (n,h,w,c)
        return self.proj2(x) + inputs

class Decoder(keras.Sequential):
    def __init__(self, img_height, img_width, name=None, download_weights=True):
        super().__init__(
            [
                keras.layers.Input((img_height // 8, img_width // 8, 4)),
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(256),
                ResnetBlock(256),
                ResnetBlock(256),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                ResnetBlock(128),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ],
            name=name,
        )

class DiffusionModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length,512), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )
        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)
        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)
        super().__init__([latent, t_embed_input, context], output, name=name)
class ImageEncoder(keras.Sequential):
    def __init__(self, download_weights=True):
        super().__init__(
            [
                keras.layers.Input((None, None, 3)),
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),
                PaddedConv2D(8, 1),
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )

class NoiseScheduler:
    def __init__(
        self,
        train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
        clip_sample=True,
    ):
        self.train_timesteps = train_timesteps

        if beta_schedule == "linear":
            self.betas = ops.linspace(beta_start, beta_end, train_timesteps)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                ops.linspace(beta_start**0.5, beta_end**0.5, train_timesteps)
                ** 2
            )
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = ops.cumprod(self.alphas)

        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.seed_generator = random.SeedGenerator(seed=42)

    def _get_variance(self, timestep, predicted_variance=None):
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        )

        variance = (
            (1 - alpha_prod_prev) / (1 - alpha_prod) * self.betas[timestep]
        )

        if self.variance_type == "fixed_small":
            variance = ops.clip(variance, x_min=1e-20, x_max=1)
        elif self.variance_type == "fixed_small_log":
            variance = ops.log(ops.clip(variance, x_min=1e-20, x_max=1))
        elif self.variance_type == "fixed_large":
            variance = self.betas[timestep]
        elif self.variance_type == "fixed_large_log":
            variance = ops.log(self.betas[timestep])
        elif self.variance_type == "learned":
            return predicted_variance
        elif self.variance_type == "learned_range":
            min_log = variance
            max_log = self.betas[timestep]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        else:
            raise ValueError(f"Invalid variance type: {self.variance_type}")

        return variance

    def step(
        self,
        model_output,
        timestep,
        sample,
        predict_epsilon=True,
    ):
        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = ops.split(
                model_output, sample.shape[1], axis=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        )
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf  # noqa: E501
        if predict_epsilon:
            pred_original_sample = (
                sample - beta_prod ** (0.5) * model_output
            ) / alpha_prod ** (0.5)
        else:
            pred_original_sample = model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = ops.clip_by_value(
                pred_original_sample, -1, 1
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current
        # sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_prev ** (0.5) * self.betas[timestep]
        ) / beta_prod
        current_sample_coeff = (
            self.alphas[timestep] ** (0.5) * beta_prod_prev / beta_prod
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if timestep > 0:
            noise = random.normal(model_output.shape, seed=self.seed_generator)
            variance = (
                self._get_variance(
                    timestep, predicted_variance=predicted_variance
                )
                ** 0.5
            ) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
        self,
        original_samples,
        noise,
        timesteps,
    ):
        sqrt_alpha_prod = ops.take(self.alphas_cumprod, timesteps) ** 0.5
        sqrt_one_minus_alpha_prod = (
            1 - ops.take(self.alphas_cumprod, timesteps)
        ) ** 0.5

        for _ in range(3):
            sqrt_alpha_prod = ops.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = ops.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )
        sqrt_alpha_prod = ops.cast(
            sqrt_alpha_prod, dtype=original_samples.dtype
        )
        sqrt_one_minus_alpha_prod = ops.cast(
            sqrt_one_minus_alpha_prod, dtype=noise.dtype
        )
        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def __len__(self):
        return self.train_timesteps

MAX_PROMPT_LENGTH = 77
class StableDiffusionBase:
   
    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width
        self._image_encoder = None
        self._text_encoder = None
        self._diffusion_model = None
        self._decoder = None
        self._tokenizer = None
        self.jit_compile = jit_compile
    def text_to_image(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        seed=None,
    ):
        encoded_text = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
        )

    def encode_text(self, prompt):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = ops.convert_to_tensor([phrase], dtype="int32")

        context = self.text_encoder.predict_on_batch(
            {"tokens": phrase, "positions": self._get_pos_ids()}
        )

        return context

    def generate_image(
        self,
        encoded_text,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
    ):
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        context = self._expand_tensor(encoded_text, batch_size)

        if negative_prompt is None:
            unconditional_context = ops.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )

        if diffusion_noise is not None:
            diffusion_noise = ops.squeeze(diffusion_noise)
            if len(ops.shape(diffusion_noise)) == 3:
                diffusion_noise = ops.repeat(
                    ops.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        num_timesteps = 1000
        ratio = (
            (num_timesteps - 1) / (num_steps - 1)
            if num_steps > 1
            else num_timesteps
        )
        timesteps = (np.arange(0, num_steps) * ratio).round().astype(np.int64)

        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": unconditional_context,
                }
            )
            latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": context,
                }
            )
            latent = ops.array(
                unconditional_latent
                + unconditional_guidance_scale * (latent - unconditional_latent)
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            # Keras backend array need to cast explicitly
            target_dtype = latent_prev.dtype
            latent = ops.cast(latent, target_dtype)
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(
                a_t
            )
            latent = (
                ops.array(latent) * math.sqrt(1.0 - a_prev)
                + math.sqrt(a_prev) * pred_x0
            )
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_unconditional_context(self):
        unconditional_tokens = ops.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS],
            dtype="int32",
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            {"tokens": unconditional_tokens, "positions": self._get_pos_ids()}
        )

        return unconditional_context

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch
        size."""
        text_embedding = ops.squeeze(text_embedding)
        if len(text_embedding.shape) == 2:
            text_embedding = ops.repeat(
                ops.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    @property
    def image_encoder(self):
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder()
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self._image_encoder

    @property
    def text_encoder(self):
        pass

    @property
    def diffusion_model(self):
        pass

    @property
    def decoder(self):
       
        if self._decoder is None:
            self._decoder = Decoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._decoder.compile(jit_compile=True)
        return self._decoder

    @property
    def tokenizer(self):
       
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def _get_timestep_embedding(
        self, timestep, batch_size, dim=320, max_period=10000
    ):
        half = dim // 2
        range = ops.cast(ops.arange(0, half), "float32")
        freqs = ops.exp(-math.log(max_period) * range / half)
        args = ops.convert_to_tensor([timestep], dtype="float32") * freqs
        embedding = ops.concatenate([ops.cos(args), ops.sin(args)], 0)
        embedding = ops.reshape(embedding, [1, -1])
        return ops.repeat(embedding, batch_size, axis=0)

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        return random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4),
            seed=seed,
        )
    @staticmethod
    def _get_pos_ids():
        return ops.expand_dims(ops.arange(MAX_PROMPT_LENGTH, dtype="int32"), 0)

class StableDiffusion(StableDiffusionBase):
    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        super().__init__(img_height, img_width, jit_compile)
    @property
    def text_encoder(self):
       
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
       
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model

# BPE合并规则是字节对编码算法中的核心部分，它决定了在词汇表构建过程中如何合并字符对以生成新的子词单元。
# BPE合并规则基于语料库中字符对（或标记对）的出现频率，通过迭代地合并最频繁出现的字符对来逐步构建词汇表。
# 具体来说，BPE合并规则的工作流程如下：
# 初始化词汇表：首先，将语料库中的所有唯一字符（或更常见的是，所有单独字符加上一个特殊的结束符号，如</w>，
# 用于标记单词的结束）作为初始词汇表。计算字符对频率：然后，遍历语料库中的单词，将它们拆分成字符序列（或预
# 先拆分成标记序列，如果使用了更复杂的预标记化步骤），并计算每个相邻字符对（或标记对）的出现频率。
# 合并最频繁字符对：找到出现频率最高的字符对，并将其合并为一个新的子词单元。这个新单元随后被添加到词汇表中
# ，并在语料库中替换所有出现的该字符对。
# 重复合并过程：继续上述过程，不断合并新的最频繁字符对，直到达到预定的词汇表大小或其他停止条件（如达到一定的合
# 并次数）。构建最终词汇表：经过多次合并后，得到包含多个子词单元的词汇表，这些子词单元可以用于后续的文本分词和模
# 型训练。BPE合并规则的关键在于其迭代合并的特性，这使得算法能够自动发现语料库中的常见字符模式，并将它们组合成更有
# 意义的子词单元。这种特性使得BPE在自然语言处理任务中特别有用，尤其是在处理罕见词、新词和跨语言文本时。
# 此外，值得注意的是，BPE算法本身并不直接处理字节，而是处理字符或标记（在预标记化步骤之后）。尽管其名称中包含“Byte”
# ，但这主要是为了与历史上的字节对编码压缩算法相区分。在现代自然语言处理应用中，BPE通常被用作一种子词级分词算法。

class SimpleTokenizer:
    def __init__(self, bpe_path=None):
        self.byte_encoder = bytes_to_unicode() # 索引-->字符
        # 字符-->索引
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values()) # 字符表
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges: 
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.vocab = vocab
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.special_tokens = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = self._create_pat()

    def _create_encoder(self, vocab):
        return dict(zip(vocab, range(len(vocab))))

    def _create_decoder(self, encoder):
        return {v: k for k, v in encoder.items()}
    # _create_pat(self): 创建一个正则表达式模式（pat），用于匹配特殊标记、单引号缩写、字母、
    # 数字以及非空白非字母非数字的字符序列。这个模式在分词过程中用于识别需要被处理的文本单元。
    def _create_pat(self):
        return re.compile(
            "|".join([re.escape(key) for key in self.special_tokens.keys()])
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    @property
    def end_of_text(self):
        return self.encoder["<|endoftext|>"]

    @property
    def start_of_text(self):
        return self.encoder["<|startoftext|>"]

    def add_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens_added = 0
        for token in tokens:
            if token in self.vocab:
                continue
            tokens_added += 1
            self.vocab.append(token)
            self.special_tokens[token] = token
            self.cache[token] = token
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        self.pat = self._create_pat()
        return tokens_added

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(" ")
            )
        return [self.start_of_text] + bpe_tokens + [self.end_of_text]

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

class DiffusionModelV2(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 1024), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)
