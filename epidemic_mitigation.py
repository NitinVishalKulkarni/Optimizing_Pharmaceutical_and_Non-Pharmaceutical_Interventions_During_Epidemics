# Imports
import sys
from collections import deque
import gym
from gym import spaces
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import time
import logging
from datetime import datetime
import pathlib


# This ensures that all the data isn't loaded into the GPU memory at once.
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Disables eager execution.
tf.compat.v1.disable_eager_execution()

# Disables Tensorflow messages.
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.experimental.output_all_intermediates(True)


# Defining the Disease Mitigation Environment.
# noinspection DuplicatedCode
class DiseaseMitigation(gym.Env):
    """This class implements the Disease Mitigation environment."""

    def __init__(self, state_name, state_population, start_date):
        """This method initializes the environment.

        :param state_name: String - Name of the state whose data we will train our model on.
        :param state_population: Integer - Population of the state we will train our model on.
        :param start_date: String - Date from which we will simulate our model. Format should be mm-dd-yyyy / yyyy-mm-dd
        ."""

        self.covid_data = pd.read_csv(f'./{state_name}.csv')
        self.covid_data['date'] = pd.to_datetime(self.covid_data['date'])
        self.population = state_population
        self.start_date = start_date
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(12)

        # Population Dynamics by Epidemiological Compartments:
        self.number_of_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible'].iloc[0]
        self.number_of_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed'].iloc[0]
        self.number_of_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected'].iloc[0]
        self.number_of_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized'].iloc[0]
        self.number_of_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered'].iloc[0]
        self.number_of_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased'].iloc[0]

        # Population Dynamics by Vaccination Status:
        self.number_of_unvaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'unvaccinated_individuals'].iloc[0]
        self.number_of_fully_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'fully_vaccinated_individuals'].iloc[0]
        self.number_of_booster_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'boosted_individuals'].iloc[0]

        # Susceptible Compartment by Vaccination Status:
        self.number_of_unvaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_UV'].iloc[0]
        self.number_of_fully_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_FV'].iloc[0]
        self.number_of_booster_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_BV'].iloc[0]

        # Exposed Compartment by Vaccination Status:
        self.number_of_unvaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_UV'].iloc[0]
        self.number_of_fully_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_FV'].iloc[0]
        self.number_of_booster_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_BV'].iloc[0]

        # Infected Compartment by Vaccination Status:
        self.number_of_unvaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_UV'].iloc[0]
        self.number_of_fully_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_FV'].iloc[0]
        self.number_of_booster_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_BV'].iloc[0]

        # Hospitalized Compartment by Vaccination Status:
        self.number_of_unvaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_UV'].iloc[0]
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_FV'].iloc[0]
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_BV'].iloc[0]

        # Recovered Compartment by Vaccination Status:
        self.number_of_unvaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_UV'].iloc[0]
        self.number_of_fully_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_FV'].iloc[0]
        self.number_of_booster_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_BV'].iloc[0]

        # Deceased Compartment by Vaccination Status:
        self.number_of_unvaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_UV'].iloc[0]
        self.number_of_fully_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_FV'].iloc[0]
        self.number_of_booster_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_BV'].iloc[0]

        # LISTS FOR PLOTTING:
        # Lists by Epidemiological Compartments:
        self.number_of_susceptible_individuals_list = [self.number_of_susceptible_individuals]
        self.number_of_exposed_individuals_list = [self.number_of_exposed_individuals]
        self.number_of_infected_individuals_list = [self.number_of_infected_individuals]
        self.number_of_hospitalized_individuals_list = [self.number_of_hospitalized_individuals]
        self.number_of_recovered_individuals_list = [self.number_of_recovered_individuals]
        self.number_of_deceased_individuals_list = [self.number_of_deceased_individuals]

        # Lists by Vaccination Status:
        self.number_of_unvaccinated_individuals_list = [self.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = [self.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = [self.number_of_booster_vaccinated_individuals]

        # Lists for Epidemiological Compartments by Vaccination Status:
        # Susceptible Compartment
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.number_of_booster_vaccinated_susceptible_individuals]

        # Exposed Compartment
        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.number_of_booster_vaccinated_exposed_individuals]

        # Infected Compartment
        self.number_of_unvaccinated_infected_individuals_list = \
            [self.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.number_of_booster_vaccinated_infected_individuals]

        # Hospitalized Compartment
        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.number_of_booster_vaccinated_hospitalized_individuals]

        # Recovered Compartment
        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.number_of_booster_vaccinated_recovered_individuals]

        # Deceased Compartment
        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.number_of_booster_vaccinated_deceased_individuals]

        self.economic_and_social_rate = 100.0
        self.economic_and_social_rate_list = [self.economic_and_social_rate]

        # Values of the epidemiological model parameters:
        self.alpha = [0.9921061737800656, 0.999999999999382, 0.9997006558362771, 0.9968623277813831,
                      0.9999999976899998, 0.9999999999934777, 0.9999832708276221, 0.9999977204877676,
                      0.999990484796472, 0.9999998603343343, 0.9999994613134795, 0.8667971932071317,
                      0.9987829564173087, 0.9242816529512278, 0.8913601714594757]

        # Exposure Rates
        self.beta_original = [4.979203079794559, 4.97458653699875, 4.998978463651875, 4.150487813293975,
                              4.999575013079198, 4.99997112741788, 4.878199450296301, 4.902405795261032,
                              4.999999999999789, 3.3489137022430944, 4.999972601990617, 3.6779583437607575,
                              1.6469992682905124, 2.129861260361653, 4.204429178608633]
        self.beta = None

        # Hospitalization Rates:
        self.delta_uv = [0.0025502796914869566, 1.1927977532355527e-14, 0.00033590586973677593, 0.004439027683949327,
                         0.004444439879977823, 0.004438688228942953, 0.003410784678446129, 0.0019063365305083818,
                         0.002738613153107736, 0.0031973457209689992, 0.0012563474592947861, 0.0015391354807226135,
                         0.0003191149877727051, 0.0028976182507645257, 0.0009689996872575844]

        self.delta_fv = [0.0013644550830824043, 7.998410292233626e-10, 0.004441947264565481, 0.00310033406963474,
                         0.003111528871142363, 0.0021878944815369207, 0.003842754738498646, 3.0471233519136476e-05,
                         0.0008992145473468121, 0.000546659225923523, 0.0004165955972560249, 0.00040581377314839235,
                         0.0012433423734815078, 0.0029025653261004203, 0.0013589929204257342]

        self.delta_bv = [0.000516666, 0.000516666, 0.000516666, 0.000516666, 0.000516666, 0.000516666,
                         0.004435513215501976, 0.0008285269025012719, 0.0010351714630686705, 0.00035536520680206913,
                         0.000436249724771229, 0.0003222169211899946, 0.00028863269709794153, 0.0011291977574560646,
                         0.0004754980986263272]

        # Recovery Rates:
        self.gamma_i_uv = [0.05454324458277713, 0.054999999999984624, 0.05493914662841438, 0.05499990759397898,
                           0.05499806747662576, 0.04100138873053479, 0.05477290137597402, 0.053825419002336575,
                           0.054529234511264964, 0.04000000000000034, 0.046392374251813036, 0.054999713367516335,
                           0.04000004218536563, 0.0400000026033707, 0.04012178719368275]

        self.gamma_i_fv = [0.05499214815619127, 0.051716264016610655, 0.05496900364583629, 0.05461994710842431,
                           0.05499999997555863, 0.04747274559193559, 0.05326993709415978, 0.054999468572432986,
                           0.05499999748127864, 0.045000001732971695, 0.052260588208621334, 0.054998278481102426,
                           0.045000030630856995, 0.05443487833553049, 0.05474162345518843]

        self.gamma_i_bv = [0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.047527421385129165, 0.047512182262614014,
                           0.04753550670362312, 0.047500000000000084, 0.06175487524141818, 0.06499802339400954,
                           0.047500619893717684, 0.04753664736862204, 0.06291341416198146]

        self.gamma_h_uv = [0.03467454100259981, 0.025387756477381746, 0.025004448376817784, 0.02761422403317452,
                           0.025000000000236458, 0.05251118105209822, 0.0549110466748263, 0.033067036304681066,
                           0.025000016934706253, 0.05499999999986076, 0.04197826269230298, 0.04834864909249086,
                           0.026320057827536415, 0.02500001559861162, 0.025028212484802824]

        self.gamma_h_fv = [0.030290563140145815, 0.030000000000000207, 0.03185761802081461, 0.04204778355488563,
                           0.05354273969484376, 0.054191972978106906, 0.04554706163804386, 0.030000011314920128,
                           0.03038156224719357, 0.05499999999999999, 0.04614880587376313, 0.04269476504178644,
                           0.03418548843830638, 0.030676406640121834, 0.03000935720574363]

        self.gamma_h_bv = [0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.05024106727480235,
                           0.06122977561546748, 0.03339888791458119, 0.030000018417240464, 0.05951600677771558,
                           0.04358177542024513, 0.03491646714746562, 0.06491527721233392, 0.04218485117522834]

        # Death Rates:
        self.mu_i_uv = [0.0011898381320066536, 0.0016461696490861401, 5.557756084789282e-05, 0.0025929773893008544,
                        0.003332580613624076, 0.00038434227102280644, 0.0014475754401885475, 0.0006446404408645657,
                        0.00045090847272443433, 5.5555550000155024e-05, 0.0002842322533048449, 0.0033331446010170136,
                        0.0001346545061541397, 0.00016510386851820654, 0.001174139354820452]

        self.mu_i_fv = [0.000968991153713453, 0.0008031958184529309, 5.555555000134667e-06, 0.0007799791561391193,
                        0.00012356244610305732, 0.00028252658788236753, 0.0009695630490258248, 0.0007025163121404912,
                        0.0005389402123607347, 0.00022287710504662042, 0.0003852112799188121, 0.0028162301234761593,
                        1.658517276642031e-05, 0.00012902217569051903, 0.00020343028748589007]

        self.mu_i_bv = [5.555555000000009e-06, 5.555555000000009e-06, 5.555555000000009e-06, 5.555555000000009e-06,
                        5.555555000000009e-06, 5.555555000000009e-06, 0.00023056307212302528, 0.00016819448709593123,
                        3.36418613048688e-05, 0.0002482277445804881, 0.0003162945618154334, 0.0012679416366859486,
                        0.000820640351487956, 1.709677015416232e-05, 0.00017848713214707823]

        self.mu_h_uv = [0.0065387373871696065, 0.0027777723418094774, 0.0027777707610357793, 0.0027796898070118975,
                        0.002908950156740234, 0.002813558388157619, 0.007375748874639409, 0.0027833913951630404,
                        0.002781769011272498, 0.011425345371481198, 0.007260659510285804, 0.011103496703024114,
                        0.0027786870476711072, 0.0027777753029138794, 0.002778499268413278]

        self.mu_h_fv = [0.0007779467358999902, 0.0036787721921557924, 0.012782730610475765, 0.0007780821739797155,
                        0.010918170242898977, 0.01065527161079136, 0.013786555042614205, 0.0007777747122808551,
                        0.000782712451470584, 0.013888879999999729, 0.006634152071829002, 0.007622872559056293,
                        0.005436367402536328, 0.0014393769862645256, 0.0007832171167080535]

        self.mu_h_bv = [0.0008777700000000002, 0.0008777700000000002, 0.0008777700000000002, 0.0008777700000000002,
                        0.0008777700000000002, 0.0008777700000000002, 0.00978766881560744, 0.013888851250363472,
                        0.0073671130602453875, 0.01388887999999899, 0.013887747182280856, 0.01253248402718307,
                        0.0033045026940641932, 0.013875288010646254, 0.0020271548979584873]

        # Exposure to Susceptible and Recovered Rates:
        self.sigma_s_uv = [0.2797044028880083, 0.34431409936925694, 0.3580504182316429, 0.1166177252469825,
                           0.16997197499388433, 0.23405769641743218, 0.2769261712186397, 0.2566061899692816,
                           0.174547479075425, 0.0, 0.15547516429999247, 0.02161976137750099, 0.00810782879033839,
                           0.0018613327530079271, 2.7699159632632586e-10]

        self.sigma_s_fv = [0.2662392505973987, 0.33212344139038685, 0.3261725159472545, 0.1490687843082517,
                           0.17262913177692407, 0.2511983728144259, 0.29030016577761925, 0.26604931009425203,
                           0.19978065051003357, 0.019989520104101544, 0.29600857702146344, 0.09553721531017201,
                           0.03525179288803204, 0.013619697748734672, 0.2665718014138282]

        self.sigma_s_bv = [0.5, 0.5, 0.5, 0.5, 0.5, 0.14985202566588945, 0.21973900778289351, 0.2599201628582599,
                           0.20235521660244826, 0.021243110081864858, 0.3154426026890688, 0.07282721934828185,
                           0.06104703495295127, 0.014707636143268588, 0.23840873738908486]

        self.sigma_r_uv = [0.03953448795218578, 0.058203388178277304, 0.032928459326050485, 0.0515539619520361,
                           0.034443635234412906, 0.059315200907886834, 0.07371744803997626, 0.08680504616510387,
                           0.09303842995851996, 0.09084809679551148, 0.9042612659047722, 0.7424077412987622,
                           0.9062605142439981, 0.9961753414307617, 3.271895679946013e-07]

        self.sigma_r_fv = [0.019437664121463194, 0.021335162733075785, 0.029657269748876447, 0.011550470005043612,
                           0.022108016565385413, 0.026750229515834056, 0.027068151213753, 0.032421554164926925,
                           0.02804895096106269, 0.018719326095313793, 0.06399355798909162, 0.03261880318624616,
                           0.03236626547518978, 0.0027465977594754443, 0.011397488403402434]

        self.sigma_r_bv = [0.5, 0.5, 0.5, 0.5, 0.5, 0.36269708395031774, 0.0765039340194445, 0.04085645680385164,
                           0.0319032123657676, 0.01950260146638827, 0.0672133875063915, 0.05921676760645461,
                           0.00578248066924314, 0.0016780890594801368, 0.03637156804177161]

        # Infection Rates:
        self.zeta_s_uv = [0.0007318991017170318, 0.0011574000234550435, 0.0017036948546286818, 0.004362887453730158,
                          0.013457377607078725, 0.006346703985227398, 0.011831358692383566, 0.01252818896039103,
                          0.020312861456003757, 0.033087401587158706, 0.01400141647252155, 0.0008811232498273897,
                          0.049999988741923546, 0.04999999994431923, 6.647410305418711e-06]

        self.zeta_s_fv = [6.1106906552897415e-06, 0.00013962269248590849, 0.00027731392827476555,
                          0.0008033915116582933, 0.0013398147739391419, 0.0006858134315300495, 0.00034256391767068574,
                          0.0003035016049242131, 0.00046329586694962636, 0.0011518632138663876, 5.289629342576374e-07,
                          2.5765181231485192e-08, 0.00015346896176303293, 0.00031249562807744675, 1.929435442860061e-09]

        self.zeta_s_bv = [0.0003000000000000001, 0.0003000000000000001, 0.0003000000000000001, 0.0003000000000000001,
                          0.0003000000000000001, 0.001149491404118182, 0.004036702171532546, 3.397575846220136e-10,
                          0.0001682986024646966, 0.000999212015626945, 0.00019274732954734208, 5.880573818801605e-05,
                          0.0009881243669758186, 0.001658550667957768, 0.0005631708299375719]

        self.zeta_r_uv = [0.0024909333130088587, 0.0007187527866670652, 0.00023545233240417074, 0.0057906673455601744,
                          6.310338210774314e-07, 0.0011364601214976788, 0.00063194566537908, 1.2389820280844788e-10,
                          2.420593336882604e-09, 0.0033726612629533664, 8.856118519284806e-06, 5.118492442954259e-06,
                          0.04999942155546967, 0.049999999998551835, 1.127830620561987e-08]

        self.zeta_r_fv = [0.0006450657128552588, 1.2115125569422958e-13, 2.265172591711384e-06, 0.00021820919238626935,
                          1.872674237901606e-12, 4.2976049414082955e-06, 2.0901160795103735e-06, 6.318968785489765e-07,
                          1.0528939442533414e-10, 2.8798764222455142e-05, 0.0003786947524119257, 1.621949413471713e-08,
                          0.00022645875956498336, 0.000495940501522658, 0.000724824228763912]

        self.zeta_r_bv = [0.004, 0.004, 0.004, 0.004, 0.004, 0.006979005796484052, 0.0010701088453985836,
                          0.0013051640745155204, 0.0006450084026508098, 0.0010040242434811648, 0.0005112768916504632,
                          7.871312454313601e-05, 0.0009921237117573275, 0.0015537442266259076, 0.002874814750625074]

        # Hyperparameters for reward function.
        self.economic_and_social_rate_lower_limit = 70
        self.economic_and_social_rate_coefficient = 1
        self.infection_coefficient = 500_000
        self.penalty_coefficient = 1_000
        self.deceased_coefficient = 10_000

        self.max_timesteps = 181
        self.timestep = 0

        # To help avoid rapidly changing policies.
        self.action_history = []
        self.previous_action = 0
        self.current_action = 0

        self.allowed_actions = [True, True, True, True, True]
        self.required_actions = [False, False, False, False, False]
        self.allowed_actions_numbers = [1 for _ in range(self.action_space.n)]

        self.min_no_npm_pm_period = 14
        self.min_sdm_period = 28
        self.min_lockdown_period = 14
        self.min_mask_mandate_period = 28
        self.min_vaccination_mandate_period = 0

        self.max_no_npm_pm_period = 56
        self.max_sdm_period = 112
        self.max_lockdown_period = 42
        self.max_mask_mandate_period = 180
        self.max_vaccination_mandate_period = 0

        self.no_npm_pm_counter = 0
        self.sdm_counter = 0
        self.lockdown_counter = 0
        self.mask_mandate_counter = 0
        self.vaccination_mandate_counter = 0

        self.new_cases = []

    def reset(self):
        """This method resets the environment and returns the state as the observation.

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)"""

        # Population Dynamics by Epidemiological Compartments:
        self.number_of_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible'].iloc[0]
        self.number_of_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed'].iloc[0]
        self.number_of_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected'].iloc[0]
        self.number_of_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized'].iloc[0]
        self.number_of_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered'].iloc[0]
        self.number_of_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased'].iloc[0]

        # Population Dynamics by Vaccination Status:
        self.number_of_unvaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'unvaccinated_individuals'].iloc[0]
        self.number_of_fully_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'fully_vaccinated_individuals'].iloc[0]
        self.number_of_booster_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'boosted_individuals'].iloc[0]

        # Susceptible Compartment by Vaccination Status:
        self.number_of_unvaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_UV'].iloc[0]
        self.number_of_fully_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_FV'].iloc[0]
        self.number_of_booster_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_BV'].iloc[0]

        # Exposed Compartment by Vaccination Status:
        self.number_of_unvaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_UV'].iloc[0]
        self.number_of_fully_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_FV'].iloc[0]
        self.number_of_booster_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_BV'].iloc[0]

        # Infected Compartment by Vaccination Status:
        self.number_of_unvaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_UV'].iloc[0]
        self.number_of_fully_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_FV'].iloc[0]
        self.number_of_booster_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_BV'].iloc[0]

        # Hospitalized Compartment by Vaccination Status:
        self.number_of_unvaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_UV'].iloc[0]
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_FV'].iloc[0]
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_BV'].iloc[0]

        # Recovered Compartment by Vaccination Status:
        self.number_of_unvaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_UV'].iloc[0]
        self.number_of_fully_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_FV'].iloc[0]
        self.number_of_booster_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_BV'].iloc[0]

        # Deceased Compartment by Vaccination Status:
        self.number_of_unvaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_UV'].iloc[0]
        self.number_of_fully_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_FV'].iloc[0]
        self.number_of_booster_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_BV'].iloc[0]

        # LISTS FOR PLOTTING:
        # Lists by Epidemiological Compartments:
        self.number_of_susceptible_individuals_list = [self.number_of_susceptible_individuals]
        self.number_of_exposed_individuals_list = [self.number_of_exposed_individuals]
        self.number_of_infected_individuals_list = [self.number_of_infected_individuals]
        self.number_of_hospitalized_individuals_list = [self.number_of_hospitalized_individuals]
        self.number_of_recovered_individuals_list = [self.number_of_recovered_individuals]
        self.number_of_deceased_individuals_list = [self.number_of_deceased_individuals]

        # Lists by Vaccination Status:
        self.number_of_unvaccinated_individuals_list = [self.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = [self.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = [self.number_of_booster_vaccinated_individuals]

        # Lists for Epidemiological Compartments by Vaccination Status:
        # Susceptible Compartment
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.number_of_booster_vaccinated_susceptible_individuals]

        # Exposed Compartment
        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.number_of_booster_vaccinated_exposed_individuals]

        # Infected Compartment
        self.number_of_unvaccinated_infected_individuals_list = \
            [self.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.number_of_booster_vaccinated_infected_individuals]

        # Hospitalized Compartment
        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.number_of_booster_vaccinated_hospitalized_individuals]

        # Recovered Compartment
        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.number_of_booster_vaccinated_recovered_individuals]

        # Deceased Compartment
        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.number_of_booster_vaccinated_deceased_individuals]

        self.economic_and_social_rate = 100
        self.economic_and_social_rate_list = [self.economic_and_social_rate]

        self.new_cases = []

        self.timestep = 0

        """To help avoid rapidly changing policies."""
        self.action_history = []
        self.previous_action = 0
        self.current_action = 0

        # Counter to keep track of the consecutive times an action was taken.
        self.no_npm_pm_counter = 0
        self.sdm_counter = 0
        self.lockdown_counter = 0
        self.mask_mandate_counter = 0
        self.vaccination_mandate_counter = 0

        # Boolean to check whether an action is allowed.
        no_npm_pm_allowed = True
        sdm_allowed = True
        lockdown_allowed = True
        mask_mandate_allowed = True
        vaccination_mandate_allowed = True

        no_npm_pm_required, sdm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required = \
            False, False, False, False, False

        self.allowed_actions = [no_npm_pm_allowed, sdm_allowed, lockdown_allowed,
                                mask_mandate_allowed, vaccination_mandate_allowed]
        self.required_actions = \
            [no_npm_pm_required, sdm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required]
        self.allowed_actions_numbers = [1 for _ in range(self.action_space.n)]

        economic_and_social_rate_lower_limit_violated = False

        index = 0

        min_lockdown_duration_penalty, min_mask_mandate_duration_penalty, min_vaccination_mandate_penalty = \
            False, False, False
        max_lockdown_duration_penalty, max_mask_mandate_duration_penalty = False, False

        observation = \
            [self.number_of_exposed_individuals / self.population,
             self.number_of_infected_individuals / self.population,
             self.number_of_deceased_individuals / self.population,
             self.number_of_unvaccinated_individuals / self.population,
             self.number_of_fully_vaccinated_individuals / self.population,
             self.number_of_booster_vaccinated_individuals / self.population,
             self.economic_and_social_rate / 100, economic_and_social_rate_lower_limit_violated,
             min_lockdown_duration_penalty, min_mask_mandate_duration_penalty, min_vaccination_mandate_penalty,
             max_lockdown_duration_penalty, max_mask_mandate_duration_penalty,
             no_npm_pm_allowed, lockdown_allowed, mask_mandate_allowed, vaccination_mandate_allowed,
             no_npm_pm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required,
             self.no_npm_pm_counter, self.lockdown_counter, self.mask_mandate_counter,
             self.vaccination_mandate_counter, index, self.previous_action]

        # Simpler observation:
        observation = \
            [self.number_of_infected_individuals / self.population,
             self.economic_and_social_rate / 100, self.previous_action, self.current_action]

        return observation

    def step(self, action):
        """This method implements what happens when the agent takes a particular action. It changes the rate at which
        new people are infected, defines the rewards for the various states, and determines when the episode ends.

        :param action: - Integer in the range 0 to 1 inclusive.

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)
                 reward: - (Float value that's used to measure the performance of the agent.)
                 done: - (Boolean describing whether the episode has ended.)
                 info: - (A dictionary that can be used to provide additional implementation information.)"""

        self.action_history.append(action)

        if len(self.action_history) == 1:
            self.previous_action = 0
        else:
            self.previous_action = self.action_history[-2]
        self.current_action = action

        index = int(np.floor((self.timestep + 214) / 28))

        # Updating the action dependent parameters:
        if action == 0:  # No NPM or PM taken. 7.3
            self.beta = self.beta_original[index] * 1.4 \
                if self.number_of_infected_individuals / self.population >= 0.001 \
                else self.beta_original[index] * 1.1
            self.economic_and_social_rate = min(1.005 * self.economic_and_social_rate, 100) \
                if self.number_of_infected_individuals / self.population < 0.001 \
                else 0.999 * self.economic_and_social_rate
            self.no_npm_pm_counter += 1
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 1:  # SDM
            self.beta = self.beta_original[index] * 0.95
            self.economic_and_social_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 2:  # Lockdown (Closure of non-essential business, schools, gyms...) 0.997
            self.beta = self.beta_original[index] * 0.85
            self.economic_and_social_rate *= 0.997
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 3:  # Public Mask Mandates 0.9975
            self.beta = self.beta_original[index] * 0.925
            self.economic_and_social_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 4:  # Vaccination Mandates 0.994
            self.beta = self.beta_original[index] * 0.95
            self.economic_and_social_rate *= 0.994
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 5:  # SDM and Public Mask Mandates 0.9965
            self.beta = self.beta_original[index] * 0.875
            self.economic_and_social_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 6:  # SDM and Vaccination Mandates 0.993
            self.beta = self.beta_original[index] * 0.825
            self.economic_and_social_rate *= 0.993
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 7:  # Lockdown and Public Mask Mandates 0.9965
            self.beta = self.beta_original[index] * 0.75
            self.economic_and_social_rate *= 0.994
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 8:  # Lockdown and Vaccination Mandates 0.993
            self.beta = self.beta_original[index] * 0.80
            self.economic_and_social_rate *= 0.993
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 9:  # Public Mask Mandates and Vaccination Mandates 0.9935
            self.beta = self.beta_original[index] * 0.90
            self.economic_and_social_rate *= 0.9935
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1
        elif action == 10:  # SDM, Public Mask Mandates and Vaccination Mandates 0.9925
            self.beta = self.beta_original[index] * 0.60
            self.economic_and_social_rate *= 0.9925
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1

        elif action == 11:  # Lockdown, Public Mask Mandates and Vaccination Mandates 0.9925
            self.beta = self.beta_original[index] * 0.60
            self.economic_and_social_rate *= 0.9925
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1

        self.compute_population_dynamics(action)
        self.economic_and_social_rate_list.append(self.economic_and_social_rate)

        # Checking which actions are allowed:
        # Potential Violations:
        no_npm_pm_min_period_violation = True if (0 < self.no_npm_pm_counter < self.min_no_npm_pm_period) else False
        sdm_min_period_violation = True if (0 < self.sdm_counter < self.min_sdm_period) else False
        lockdown_min_period_violation = True if (0 < self.lockdown_counter < self.min_lockdown_period) else False
        mask_mandate_min_period_violation = \
            True if (0 < self.mask_mandate_counter < self.min_mask_mandate_period) else False
        vaccination_mandate_min_period_violation = \
            True if (0 < self.vaccination_mandate_counter < self.min_vaccination_mandate_period) else False

        no_npm_pm_max_period_violation = True if (self.no_npm_pm_counter >= self.max_no_npm_pm_period) else False
        sdm_max_period_violation = True if (self.sdm_counter >= self.max_sdm_period) else False
        lockdown_max_period_violation = True if (self.lockdown_counter >= self.max_lockdown_period) else False
        mask_mandate_max_period_violation = True if \
            (self.mask_mandate_counter >= self.max_mask_mandate_period) else False
        vaccination_mandate_max_period_violation = \
            True if (self.vaccination_mandate_counter >= self.max_vaccination_mandate_period) else False

        # Required Actions (As in not taking them will result in minimum violation):
        no_npm_pm_required = True if no_npm_pm_min_period_violation else False
        sdm_required = True if sdm_min_period_violation else False
        lockdown_required = True if lockdown_min_period_violation else False
        mask_mandate_required = True if mask_mandate_min_period_violation else False
        vaccination_mandate_required = True if vaccination_mandate_min_period_violation else False

        # Allowed Actions
        sdm_allowed = True if ((not no_npm_pm_min_period_violation)
                               and (not lockdown_min_period_violation)
                               and (not sdm_max_period_violation)) else False

        lockdown_allowed = True if ((not no_npm_pm_min_period_violation)
                                    and (not sdm_min_period_violation)
                                    and (not lockdown_max_period_violation)) else False

        mask_mandate_allowed = True if ((not no_npm_pm_min_period_violation)
                                        and (not mask_mandate_max_period_violation)) else False

        vaccination_mandate_allowed = True if ((not no_npm_pm_min_period_violation)
                                               and (not vaccination_mandate_max_period_violation)) else False

        # No NPM and PM
        no_npm_pm_allowed = \
            True if ((not sdm_min_period_violation)
                     and (not lockdown_min_period_violation)
                     and (not mask_mandate_min_period_violation)
                     and (not vaccination_mandate_min_period_violation)
                     and (not no_npm_pm_max_period_violation)) else False
        # no_npm_pm_required = True if (no_npm_pm_min_period_violation and no_npm_pm_allowed) else False

        # # Lockdown
        # if ((not no_npm_pm_min_period_violation)
        #         and (not mask_mandate_min_period_violation)
        #         and (not vaccination_mandate_min_period_violation)
        #         and (not lockdown_max_period_violation)):
        #     lockdown_allowed = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and mask_mandate_min_period_violation
        #       and (not vaccination_mandate_min_period_violation)
        #       and (not lockdown_max_period_violation)):
        #     lockdown_allowed = True
        #     mask_mandate_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and (not mask_mandate_min_period_violation)
        #       and vaccination_mandate_min_period_violation
        #       and (not lockdown_max_period_violation)):
        #     lockdown_allowed = True
        #     vaccination_mandate_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and mask_mandate_min_period_violation
        #       and vaccination_mandate_min_period_violation
        #       and (not lockdown_max_period_violation)):
        #     lockdown_allowed = True
        #     mask_mandate_required = True
        #     vaccination_mandate_required = True
        #
        # else:
        #     lockdown_allowed = False
        #
        # # Mask Mandate
        # if ((not no_npm_pm_min_period_violation)
        #         and (not lockdown_min_period_violation)
        #         and (not vaccination_mandate_min_period_violation)
        #         and (not mask_mandate_max_period_violation)):
        #     mask_mandate_allowed = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and lockdown_min_period_violation
        #       and (not vaccination_mandate_min_period_violation)
        #       and (not mask_mandate_max_period_violation)):
        #     mask_mandate_allowed = True
        #     lockdown_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and (not lockdown_min_period_violation)
        #       and vaccination_mandate_min_period_violation
        #         and (not mask_mandate_max_period_violation)):
        #     mask_mandate_allowed = True
        #     vaccination_mandate_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and lockdown_min_period_violation
        #       and vaccination_mandate_min_period_violation
        #         and (not mask_mandate_max_period_violation)):
        #     mask_mandate_allowed = True
        #     lockdown_required = True
        #     vaccination_mandate_required = True
        #
        # else:
        #     mask_mandate_allowed = False
        #
        # # Vaccination Mandate
        # if ((not no_npm_pm_min_period_violation)
        #         and (not lockdown_min_period_violation)
        #         and (not mask_mandate_min_period_violation)
        #         and (not vaccination_mandate_max_period_violation)):
        #     vaccination_mandate_allowed = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and lockdown_min_period_violation
        #       and (not mask_mandate_min_period_violation)
        #         and (not vaccination_mandate_max_period_violation)):
        #     vaccination_mandate_allowed = True
        #     lockdown_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and (not lockdown_min_period_violation)
        #       and mask_mandate_min_period_violation
        #         and (not vaccination_mandate_max_period_violation)):
        #     vaccination_mandate_allowed = True
        #     mask_mandate_required = True
        #
        # elif ((not no_npm_pm_min_period_violation)
        #       and lockdown_min_period_violation
        #       and mask_mandate_min_period_violation
        #         and (not vaccination_mandate_max_period_violation)):
        #     vaccination_mandate_allowed = True
        #     lockdown_required = True
        #     mask_mandate_required = True
        #
        # else:
        #     vaccination_mandate_allowed = False

        # # Imposing maximum action durations:
        max_lockdown_duration_penalty = False
        max_mask_mandate_duration_penalty = False

        # Penalizing the agent for violation of the minimum action period or the economic and social rate lower limit.
        economic_and_social_rate_lower_limit_violated = \
            True if self.economic_and_social_rate < self.economic_and_social_rate_lower_limit else False

        previous_no_npm_pm_required, previous_lockdown_required,\
            previous_mask_mandate_required, previous_vaccination_mandate_required = \
            self.required_actions[0], self.required_actions[1], self.required_actions[2], self.required_actions[3]

        # penalty = True if (
        #         (previous_lockdown_required and action in [0, 2, 3, 6])
        #         or (previous_mask_mandate_required and action in [0, 1, 3, 5])
        #         or (previous_vaccination_mandate_required and action in [0, 1, 2, 4])
        # ) else False

        min_lockdown_duration_penalty = True if (
                (previous_lockdown_required and action in [0, 2, 3, 6])) else False
        min_mask_mandate_duration_penalty = True if (
                (previous_mask_mandate_required and action in [0, 1, 3, 5])) else False
        min_vaccination_mandate_penalty = True if (
                (previous_vaccination_mandate_required and action in [0, 1, 2, 4])) else False

        # Updating the lists for allowed and required actions.
        self.allowed_actions = [no_npm_pm_allowed, sdm_allowed, lockdown_allowed, mask_mandate_allowed,
                                vaccination_mandate_allowed]
        self.required_actions = [no_npm_pm_required, sdm_required, lockdown_required,
                                 mask_mandate_required, vaccination_mandate_required]

        # Logic to determine which action as per the numbers is allowed.
        action_association_list = [[0], [1, 5, 6, 10], [2, 7, 8, 11], [3, 5, 7, 9, 10, 11], [4, 6, 8, 9, 10, 11]]
        actions_allowed = None

        for i in range(5):
            if self.required_actions[i]:
                if actions_allowed is None:
                    actions_allowed = set(action_association_list[i])
                else:
                    actions_allowed = actions_allowed & set(action_association_list[i])

        for i in range(5):
            if not self.allowed_actions[i] and not self.required_actions[i]:
                if actions_allowed is None:
                    break
                else:
                    actions_allowed = actions_allowed.difference(set(action_association_list[i]))

        if actions_allowed is None:
            for i in range(5):
                if self.allowed_actions[i]:
                    if actions_allowed is None:
                        actions_allowed = set(action_association_list[i])
                    else:
                        actions_allowed = actions_allowed.union(set(action_association_list[i]))
            for i in range(5):
                if not self.allowed_actions[i]:
                    actions_allowed = actions_allowed.difference(set(action_association_list[i]))

            # actions_allowed = [i for i in range(self.action_space.n)]

        actions_allowed = list(actions_allowed)
        # print('ENV AA', actions_allowed)
        self.allowed_actions_numbers = [1 if i in actions_allowed else 0 for i in range(self.action_space.n)]

        # Reward
        reward = ((-self.infection_coefficient * self.number_of_infected_individuals / self.population)
                  + self.economic_and_social_rate)

        # if min_lockdown_duration_penalty or min_mask_mandate_duration_penalty or min_vaccination_mandate_penalty:
        #     reward = -self.penalty_coefficient / 2
        # if (max_lockdown_duration_penalty or max_mask_mandate_duration_penalty) \
        #         and not economic_and_social_rate_lower_limit_violated:
        #     reward = -self.penalty_coefficient * 7.5
        # elif (max_lockdown_duration_penalty or max_mask_mandate_duration_penalty) \
        #         and economic_and_social_rate_lower_limit_violated:
        #     reward = -self.penalty_coefficient * 10
        # if (not max_lockdown_duration_penalty and not max_mask_mandate_duration_penalty) \
        #         and economic_and_social_rate_lower_limit_violated:
        #     reward = -self.penalty_coefficient * 5

        self.timestep += 1

        observation = \
            [self.number_of_exposed_individuals / self.population,
             self.number_of_infected_individuals / self.population,
             self.number_of_deceased_individuals / self.population,
             self.number_of_unvaccinated_individuals / self.population,
             self.number_of_fully_vaccinated_individuals / self.population,
             self.number_of_booster_vaccinated_individuals / self.population,
             self.economic_and_social_rate / 100, economic_and_social_rate_lower_limit_violated,
             min_lockdown_duration_penalty, min_mask_mandate_duration_penalty, min_vaccination_mandate_penalty,
             max_lockdown_duration_penalty, max_mask_mandate_duration_penalty,
             no_npm_pm_allowed, lockdown_allowed, mask_mandate_allowed, vaccination_mandate_allowed,
             no_npm_pm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required,
             self.no_npm_pm_counter, self.lockdown_counter, self.mask_mandate_counter,
             self.vaccination_mandate_counter, index, self.previous_action]

        # Simplified observation:
        observation = \
            [self.number_of_infected_individuals / self.population,
             self.economic_and_social_rate / 100, self.previous_action, self.current_action]

        # The episode terminates when the number of infected people becomes greater than 25 % of the population.
        done = True if (self.number_of_infected_individuals >= 0.99 * self.population or
                        self.timestep >= self.max_timesteps) else False
        info = {}

        return observation, reward, done, info

    def compute_population_dynamics(self, action):
        """This method computes the action dependent population dynamics
        :parameter action: Integer - Represents the action taken by the agent."""

        # Action dependent vaccination rates.
        if action in [3, 5, 6, 7]:
            percentage_unvaccinated_to_fully_vaccinated = 0.007084760245099044
            # percentage_fully_vaccinated_to_booster_vaccinated = 0.0017285714029114
            percentage_fully_vaccinated_to_booster_vaccinated = \
                self.covid_data['percentage_fully_vaccinated_to_boosted'].iloc[self.timestep + 214]
        else:
            percentage_unvaccinated_to_fully_vaccinated = \
                self.covid_data['percentage_unvaccinated_to_fully_vaccinated'].iloc[self.timestep + 214]
            percentage_fully_vaccinated_to_booster_vaccinated = \
                self.covid_data['percentage_fully_vaccinated_to_boosted'].iloc[self.timestep + 214]

        index = int(np.floor((self.timestep + 214) / 28))

        val = 0.05
        mu, sigma = self.beta, val * self.beta
        self.beta = np.random.normal(mu, sigma, 1)

        mu, sigma = self.alpha[index], val * self.alpha[index]
        alpha = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_uv[index], val * self.delta_uv[index]
        delta_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_fv[index], val * self.delta_fv[index]
        delta_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_bv[index], val * self.delta_bv[index]
        delta_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_uv[index], val * self.gamma_i_uv[index]
        gamma_i_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_fv[index], val * self.gamma_i_fv[index]
        gamma_i_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_bv[index], val * self.gamma_i_bv[index]
        gamma_i_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_uv[index], val * self.gamma_h_uv[index]
        gamma_h_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_fv[index], val * self.gamma_h_fv[index]
        gamma_h_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_bv[index], val * self.gamma_h_bv[index]
        gamma_h_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_uv[index], val * self.mu_i_uv[index]
        mu_i_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_fv[index], val * self.mu_i_fv[index]
        mu_i_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_bv[index], val * self.mu_i_bv[index]
        mu_i_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_uv[index], val * self.mu_h_uv[index]
        mu_h_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_fv[index], val * self.mu_h_fv[index]
        mu_h_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_bv[index], val * self.mu_h_bv[index]
        mu_h_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_uv[index], val * self.sigma_s_uv[index]
        sigma_s_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_fv[index], val * self.sigma_s_fv[index]
        sigma_s_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_bv[index], val * self.sigma_s_bv[index]
        sigma_s_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_uv[index], val * self.sigma_r_uv[index]
        sigma_r_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_fv[index], val * self.sigma_r_fv[index]
        sigma_r_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_bv[index], val * self.sigma_r_bv[index]
        sigma_r_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_uv[index], val * self.zeta_s_uv[index]
        zeta_s_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_fv[index], val * self.zeta_s_fv[index]
        zeta_s_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_bv[index], val * self.zeta_s_bv[index]
        zeta_s_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_uv[index], val * self.zeta_r_uv[index]
        zeta_r_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_fv[index], val * self.zeta_r_fv[index]
        zeta_r_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_bv[index], val * self.zeta_r_bv[index]
        zeta_r_bv = np.random.normal(mu, sigma, 1)

        # Susceptible Compartment
        number_of_unvaccinated_susceptible_individuals = \
            int(self.number_of_unvaccinated_susceptible_individuals
                - (self.beta * self.number_of_unvaccinated_susceptible_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_s_uv * self.number_of_unvaccinated_exposed_individuals -
                percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_susceptible_individuals)

        number_of_fully_vaccinated_susceptible_individuals = \
            int(self.number_of_fully_vaccinated_susceptible_individuals
                - self.beta * self.number_of_fully_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + sigma_s_fv * self.number_of_fully_vaccinated_exposed_individuals +
                percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_susceptible_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_susceptible_individuals)

        number_of_booster_vaccinated_susceptible_individuals = \
            int(self.number_of_booster_vaccinated_susceptible_individuals
                - self.beta * self.number_of_booster_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + sigma_s_bv * self.number_of_booster_vaccinated_exposed_individuals +
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_susceptible_individuals)

        number_of_susceptible_individuals = \
            number_of_unvaccinated_susceptible_individuals + \
            number_of_fully_vaccinated_susceptible_individuals + \
            number_of_booster_vaccinated_susceptible_individuals

        self.number_of_unvaccinated_susceptible_individuals_list.append(
            number_of_unvaccinated_susceptible_individuals)
        self.number_of_fully_vaccinated_susceptible_individuals_list.append(
            number_of_fully_vaccinated_susceptible_individuals)
        self.number_of_booster_vaccinated_susceptible_individuals_list.append(
            number_of_booster_vaccinated_susceptible_individuals)
        self.number_of_susceptible_individuals_list.append(number_of_susceptible_individuals)

        # Exposed Compartment
        number_of_unvaccinated_exposed_individuals = \
            int(self.number_of_unvaccinated_exposed_individuals
                + self.beta * self.number_of_unvaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_unvaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
                - zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
                - sigma_s_uv * self.number_of_unvaccinated_exposed_individuals
                - sigma_r_uv * self.number_of_unvaccinated_exposed_individuals
                - percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_exposed_individuals)

        number_of_fully_vaccinated_exposed_individuals = \
            int(self.number_of_fully_vaccinated_exposed_individuals
                + self.beta * self.number_of_fully_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_fully_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                - zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                - sigma_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                - sigma_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                + percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_exposed_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_exposed_individuals)

        number_of_booster_vaccinated_exposed_individuals = \
            int(self.number_of_booster_vaccinated_exposed_individuals
                + self.beta * self.number_of_booster_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_booster_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                - zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                - sigma_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                - sigma_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                + percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_exposed_individuals)

        number_of_exposed_individuals = \
            number_of_unvaccinated_exposed_individuals + \
            number_of_fully_vaccinated_exposed_individuals + \
            number_of_booster_vaccinated_exposed_individuals

        self.number_of_unvaccinated_exposed_individuals_list.append(
            number_of_unvaccinated_exposed_individuals)
        self.number_of_fully_vaccinated_exposed_individuals_list.append(
            number_of_fully_vaccinated_exposed_individuals)
        self.number_of_booster_vaccinated_exposed_individuals_list.append(
            number_of_booster_vaccinated_exposed_individuals)
        self.number_of_exposed_individuals_list.append(number_of_exposed_individuals)

        # Infected Compartment
        number_of_unvaccinated_infected_individuals = \
            int(self.number_of_unvaccinated_infected_individuals +
                zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
                + zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
                - delta_uv * self.number_of_unvaccinated_infected_individuals -
                gamma_i_uv * self.number_of_unvaccinated_infected_individuals -
                mu_i_uv * self.number_of_unvaccinated_infected_individuals)

        number_of_fully_vaccinated_infected_individuals = \
            int(self.number_of_fully_vaccinated_infected_individuals +
                zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                + zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                - delta_fv * self.number_of_fully_vaccinated_infected_individuals -
                gamma_i_fv * self.number_of_fully_vaccinated_infected_individuals -
                mu_i_fv * self.number_of_fully_vaccinated_infected_individuals)

        number_of_booster_vaccinated_infected_individuals = \
            int(self.number_of_booster_vaccinated_infected_individuals +
                zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                + zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                - delta_bv * self.number_of_booster_vaccinated_infected_individuals -
                gamma_i_bv * self.number_of_booster_vaccinated_infected_individuals -
                mu_i_bv * self.number_of_booster_vaccinated_infected_individuals)

        self.new_cases.append(int(
            zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
            + zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
            + zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
            + zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
            + zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
            + zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals))

        number_of_infected_individuals = \
            number_of_unvaccinated_infected_individuals + \
            number_of_fully_vaccinated_infected_individuals + \
            number_of_booster_vaccinated_infected_individuals

        self.number_of_unvaccinated_infected_individuals_list.append(
            number_of_unvaccinated_infected_individuals)
        self.number_of_fully_vaccinated_infected_individuals_list.append(
            number_of_fully_vaccinated_infected_individuals)
        self.number_of_booster_vaccinated_infected_individuals_list.append(
            number_of_booster_vaccinated_infected_individuals)
        self.number_of_infected_individuals_list.append(number_of_infected_individuals)

        # Hospitalized Compartment
        number_of_unvaccinated_hospitalized_individuals = \
            int(self.number_of_unvaccinated_hospitalized_individuals +
                delta_uv * self.number_of_unvaccinated_infected_individuals -
                gamma_h_uv * self.number_of_unvaccinated_hospitalized_individuals -
                mu_h_uv * self.number_of_unvaccinated_hospitalized_individuals)

        number_of_fully_vaccinated_hospitalized_individuals = \
            int(self.number_of_fully_vaccinated_hospitalized_individuals +
                delta_fv * self.number_of_fully_vaccinated_infected_individuals -
                gamma_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals -
                mu_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals)

        number_of_booster_vaccinated_hospitalized_individuals = \
            int(self.number_of_booster_vaccinated_hospitalized_individuals +
                delta_bv * self.number_of_booster_vaccinated_infected_individuals -
                gamma_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals -
                mu_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals)

        number_of_hospitalized_individuals = \
            number_of_unvaccinated_hospitalized_individuals + \
            number_of_fully_vaccinated_hospitalized_individuals + \
            number_of_booster_vaccinated_hospitalized_individuals

        self.number_of_unvaccinated_hospitalized_individuals_list.append(
            number_of_unvaccinated_hospitalized_individuals)
        self.number_of_fully_vaccinated_hospitalized_individuals_list.append(
            number_of_fully_vaccinated_hospitalized_individuals)
        self.number_of_booster_vaccinated_hospitalized_individuals_list.append(
            number_of_booster_vaccinated_hospitalized_individuals)
        self.number_of_hospitalized_individuals_list.append(number_of_hospitalized_individuals)

        # Recovered Compartment
        number_of_unvaccinated_recovered_individuals = \
            int(self.number_of_unvaccinated_recovered_individuals
                - (self.beta * self.number_of_unvaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_uv * self.number_of_unvaccinated_exposed_individuals
                + gamma_i_uv * self.number_of_unvaccinated_infected_individuals
                + gamma_h_uv * self.number_of_unvaccinated_hospitalized_individuals
                - percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_recovered_individuals)

        number_of_fully_vaccinated_recovered_individuals = \
            int(self.number_of_fully_vaccinated_recovered_individuals
                - (self.beta * self.number_of_fully_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                + gamma_i_fv * self.number_of_fully_vaccinated_infected_individuals
                + gamma_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals
                + percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_recovered_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_recovered_individuals)

        number_of_booster_vaccinated_recovered_individuals = \
            int(self.number_of_booster_vaccinated_recovered_individuals
                - (self.beta * self.number_of_booster_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                + gamma_i_bv * self.number_of_booster_vaccinated_infected_individuals +
                gamma_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals +
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_recovered_individuals)

        number_of_recovered_individuals = \
            number_of_unvaccinated_recovered_individuals + \
            number_of_fully_vaccinated_recovered_individuals + \
            number_of_booster_vaccinated_recovered_individuals

        self.number_of_unvaccinated_recovered_individuals_list.append(
            number_of_unvaccinated_recovered_individuals)
        self.number_of_fully_vaccinated_recovered_individuals_list.append(
            number_of_fully_vaccinated_recovered_individuals)
        self.number_of_booster_vaccinated_recovered_individuals_list.append(
            number_of_booster_vaccinated_recovered_individuals)
        self.number_of_recovered_individuals_list.append(number_of_recovered_individuals)

        # Deceased Compartment
        number_of_unvaccinated_deceased_individuals = \
            int(self.number_of_unvaccinated_deceased_individuals +
                mu_i_uv * self.number_of_unvaccinated_infected_individuals +
                mu_h_uv * self.number_of_unvaccinated_hospitalized_individuals)

        number_of_fully_vaccinated_deceased_individuals = \
            int(self.number_of_fully_vaccinated_deceased_individuals +
                mu_i_fv * self.number_of_fully_vaccinated_infected_individuals +
                mu_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals)

        number_of_booster_vaccinated_deceased_individuals = \
            int(self.number_of_booster_vaccinated_deceased_individuals +
                mu_i_bv * self.number_of_booster_vaccinated_infected_individuals +
                mu_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals)

        number_of_deceased_individuals = \
            number_of_unvaccinated_deceased_individuals + \
            number_of_fully_vaccinated_deceased_individuals + \
            number_of_booster_vaccinated_deceased_individuals

        self.number_of_unvaccinated_deceased_individuals_list.append(
            number_of_unvaccinated_deceased_individuals)
        self.number_of_fully_vaccinated_deceased_individuals_list.append(
            number_of_fully_vaccinated_deceased_individuals)
        self.number_of_booster_vaccinated_deceased_individuals_list.append(
            number_of_booster_vaccinated_deceased_individuals)
        self.number_of_deceased_individuals_list.append(number_of_deceased_individuals)

        # Population Dynamics by Vaccination Status
        self.number_of_unvaccinated_individuals = \
            self.number_of_unvaccinated_individuals \
            - percentage_unvaccinated_to_fully_vaccinated * self.number_of_unvaccinated_individuals

        self.number_of_fully_vaccinated_individuals = \
            self.number_of_fully_vaccinated_individuals \
            + percentage_unvaccinated_to_fully_vaccinated * self.number_of_unvaccinated_individuals \
            - percentage_fully_vaccinated_to_booster_vaccinated * self.number_of_fully_vaccinated_individuals

        self.number_of_booster_vaccinated_individuals = \
            self.number_of_booster_vaccinated_individuals \
            + percentage_fully_vaccinated_to_booster_vaccinated * self.number_of_fully_vaccinated_individuals

        self.number_of_unvaccinated_individuals_list.append(self.number_of_unvaccinated_individuals)
        self.number_of_fully_vaccinated_individuals_list.append(self.number_of_fully_vaccinated_individuals)
        self.number_of_booster_vaccinated_individuals_list.append(self.number_of_booster_vaccinated_individuals)

        # Synchronizing the global variables with the updated local variables:
        self.number_of_unvaccinated_susceptible_individuals = \
            number_of_unvaccinated_susceptible_individuals
        self.number_of_fully_vaccinated_susceptible_individuals = \
            number_of_fully_vaccinated_susceptible_individuals
        self.number_of_booster_vaccinated_susceptible_individuals = \
            number_of_booster_vaccinated_susceptible_individuals
        self.number_of_susceptible_individuals = number_of_susceptible_individuals

        self.number_of_unvaccinated_exposed_individuals = \
            number_of_unvaccinated_exposed_individuals
        self.number_of_fully_vaccinated_exposed_individuals = \
            number_of_fully_vaccinated_exposed_individuals
        self.number_of_booster_vaccinated_exposed_individuals = \
            number_of_booster_vaccinated_exposed_individuals
        self.number_of_exposed_individuals = number_of_exposed_individuals

        self.number_of_unvaccinated_infected_individuals = \
            number_of_unvaccinated_infected_individuals
        self.number_of_fully_vaccinated_infected_individuals = \
            number_of_fully_vaccinated_infected_individuals
        self.number_of_booster_vaccinated_infected_individuals = \
            number_of_booster_vaccinated_infected_individuals
        self.number_of_infected_individuals = number_of_infected_individuals

        self.number_of_unvaccinated_hospitalized_individuals = \
            number_of_unvaccinated_hospitalized_individuals
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            number_of_fully_vaccinated_hospitalized_individuals
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            number_of_booster_vaccinated_hospitalized_individuals
        self.number_of_hospitalized_individuals = number_of_hospitalized_individuals

        self.number_of_unvaccinated_recovered_individuals = \
            number_of_unvaccinated_recovered_individuals
        self.number_of_fully_vaccinated_recovered_individuals = \
            number_of_fully_vaccinated_recovered_individuals
        self.number_of_booster_vaccinated_recovered_individuals = \
            number_of_booster_vaccinated_recovered_individuals
        self.number_of_recovered_individuals = number_of_recovered_individuals

        self.number_of_unvaccinated_deceased_individuals = \
            number_of_unvaccinated_deceased_individuals
        self.number_of_fully_vaccinated_deceased_individuals = \
            number_of_fully_vaccinated_deceased_individuals
        self.number_of_booster_vaccinated_deceased_individuals = \
            number_of_booster_vaccinated_deceased_individuals
        self.number_of_deceased_individuals = number_of_deceased_individuals

    def render(self, mode='human'):
        """This method renders the statistical graph of the population.

        :param mode: 'human' renders to the current display or terminal and returns nothing."""

        return


# noinspection DuplicatedCode
class AdvantageWeightedRegression:
    """This class implements the AWR Agent."""

    def __init__(self, environment, alternate_network=False, offline_memory_size=10_000, iterations=10):
        """This method initializes the AWR parameters, and calls the train, evaluate and render_actions methods.

        :param environment: This is the environment on which the agent will learn.
        :param alternate_network: Boolean indicating whether to use the second deeper network.
        :param offline_memory_size: Integer indicating the size of the offline replay memory.
        :param iterations: Integer indicating the number of iterations for which the agent will train."""

        # Saving the training results.
        self.date_and_time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        pathlib.Path(f'./Results/DataFrames/{self.date_and_time}').mkdir(parents=True, exist_ok=True)

        self.environment = environment  # The environment which we need the agent to solve.
        self.environment.reset()
        self.alternate_network = alternate_network  # Boolean indicating whether to use the second deeper network.
        self.offline_replay_memory_size = offline_memory_size  # This specifies the size of the offline replay memory.
        # self.offline_replay_memory = []  # Offline replay memory.
        self.offline_replay_memory = deque(maxlen=self.offline_replay_memory_size)
        self.iterations = iterations  # Number of episodes for which the agent will train.
        self.discount_factor = 0.999  # Discount factor determines the value of the future rewards.
        self.beta = 0.5  # Hyperparameter used to calculate the exponential advantage.
        self.time_period = 14  # Number of days to consider when taking an action.
        self.actor_model, self.critic_model, self.policy_model = self.neural_network()  # Creating the networks.
        self.cumulative_rewards_evaluation = []  # List containing the cumulative rewards per episode during evaluation.

        # print('NN Summary:', self.actor_model.summary(), self.critic_model.summary(), self.policy_model.summary())
        # sys.exit()

        # Lists for plotting:
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.environment.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.environment.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.environment.number_of_booster_vaccinated_susceptible_individuals]
        self.number_of_susceptible_individuals_list = [self.environment.number_of_susceptible_individuals]

        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.environment.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.environment.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.environment.number_of_booster_vaccinated_exposed_individuals]
        self.number_of_exposed_individuals_list = \
            [self.environment.number_of_exposed_individuals]

        self.number_of_unvaccinated_infected_individuals_list = \
            [self.environment.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.environment.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.environment.number_of_booster_vaccinated_infected_individuals]
        self.number_of_infected_individuals_list = \
            [self.environment.number_of_infected_individuals]

        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_booster_vaccinated_hospitalized_individuals]
        self.number_of_hospitalized_individuals_list = \
            [self.environment.number_of_hospitalized_individuals]

        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.environment.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.environment.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.environment.number_of_booster_vaccinated_recovered_individuals]
        self.number_of_recovered_individuals_list = \
            [self.environment.number_of_recovered_individuals]

        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.environment.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.environment.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.environment.number_of_booster_vaccinated_deceased_individuals]
        self.number_of_deceased_individuals_list = \
            [self.environment.number_of_deceased_individuals]

        self.number_of_unvaccinated_individuals_list = \
            [self.environment.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = \
            [self.environment.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = \
            [self.environment.number_of_booster_vaccinated_individuals]

        self.population_dynamics = {}
        self.economic_and_social_rate_list = [self.environment.economic_and_social_rate]
        self.action_history = []
        # self.train()  # Calling the train method.
        # self.evaluate()  # Calling the evaluate method.
        self.render_actions(1)  # Calling the render method.

    def neural_network(self):
        """This method builds the actor, critic and policy networks."""

        if not self.alternate_network:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.time_period, self.environment.observation_space.n))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = LSTM(512, return_sequences=True)(input_)  # Common layer 1 for the networks.
            common2 = LSTM(256, return_sequences=False)(common1)
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common2)  # Actor output.
            values = Dense(1, activation='linear')(common2)  # Critic output.

        else:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.n,))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = Dense(1024, activation='relu')(input_)  # Common layer 1 for the networks.
            common2 = Dense(512, activation='relu')(common1)  # Common layer 2 for the networks.
            common3 = Dense(256, activation='relu')(common2)  # Common layer 3 for the networks.
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common3)  # Actor output.
            values = Dense(1, activation='linear')(common3)  # Critic output.

        def custom_loss(exponential_advantage_):
            """This method defines the custom loss wrapper function that will be used by the actor model."""

            def loss_fn(y_true, y_pred):
                # Clipping y_pred so that we don't end up taking the log of 0 or 1.
                clipped_y_pred = k.clip(y_pred, 1e-8, 1 - 1e-8)
                log_probability = y_true * k.log(clipped_y_pred)
                return k.sum(-log_probability * exponential_advantage_)

            return loss_fn

        lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.96)

        # Instantiating the actor model.
        actor_model = Model(inputs=[input_, exponential_advantage], outputs=[probabilities])
        actor_model.compile(optimizer=Adam(), loss=custom_loss(exponential_advantage))
        # actor_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=custom_loss(exponential_advantage))

        # Instantiating the critic model.
        critic_model = Model(inputs=[input_], outputs=[values])
        critic_model.compile(optimizer=Adam(), loss=tf.keras.losses.Huber())

        # Instantiating the policy model.
        policy_model = Model(inputs=[input_], outputs=[probabilities])

        return actor_model, critic_model, policy_model

    def monte_carlo_returns(self):
        """This method calculates the Monte Carlo returns given a list of rewards."""

        rewards = [item[2] for item in self.offline_replay_memory]
        monte_carlo_returns = []  # List containing the Monte-Carlo returns.
        monte_carlo_return = 0
        t = 0  # Exponent by which the discount factor is raised.

        for i in range(len(self.offline_replay_memory)):

            while not self.offline_replay_memory[i][4]:  # Execute until you encounter a terminal state.

                # Equation to calculate the Monte-Carlo return.
                monte_carlo_return += self.discount_factor ** t * rewards[i]
                i += 1  # Go to the next sample.
                t += 1  # Increasing the exponent by which the discount factor is raised.

                # Condition to check whether we have reached the end of the replay memory without the episode being
                # terminated, and if so break. (This can happen with the samples at the end of the replay memory as we
                # only store the samples till we reach the replay memory size and not till we exceed it with the episode
                # being terminated.)
                if i == len(self.offline_replay_memory):
                    # If the episode hasn't terminated but you reach the end append the Monte-Carlo return to the list.
                    monte_carlo_returns.append(monte_carlo_return)

                    # Resetting the Monte-Carlo return value and the exponent to 0.
                    monte_carlo_return = 0
                    t = 0

                    break  # Break from the loop.

            # If for one of the samples towards the end we reach the end of the replay memory and it hasn't terminated,
            # we will go back to the beginning of the for loop to calculate the Monte-Carlo return for the future
            # samples if any for whom the episode hasn't terminated.
            if i == len(self.offline_replay_memory):
                continue

            # Equation to calculate the Monte-Carlo return.
            monte_carlo_return += self.discount_factor ** t * rewards[i]

            # Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of the
            # replay memory.
            monte_carlo_returns.append(monte_carlo_return)

            # Resetting the Monte-Carlo return value and the exponent to 0.
            monte_carlo_return = 0
            t = 0

        # Normalizing the returns.
        monte_carlo_returns = np.array(monte_carlo_returns)
        monte_carlo_returns = (monte_carlo_returns - np.mean(monte_carlo_returns)) / (np.std(monte_carlo_returns)
                                                                                      + 1e-08)
        monte_carlo_returns = monte_carlo_returns.tolist()

        return monte_carlo_returns

    def td_lambda_returns(self):
        """This method calculates the TD Lambda returns."""

        rewards = [item[2] for item in self.offline_replay_memory]
        next_states = [item[3] for item in self.offline_replay_memory]
        next_states = np.asarray(next_states).reshape(-1, self.environment.observation_space.n)
        next_state_values = self.critic_model.predict(next_states).flatten()
        td_lambda_returns = []  # List containing the TD Lambda returns.
        terminal_state_indices = [i for i in range(len(self.offline_replay_memory)) if self.offline_replay_memory[i][4]]
        td_n_return = 0
        t = 0  # Exponent by which the discount factor is raised.
        td_lambda_value = 0.9
        index = 0  # Pointer for keeping track of the next terminal state.
        next_terminal_state_index = terminal_state_indices[index]
        for i in range(len(self.offline_replay_memory)):
            j = i  # Used to calcuate the lambda values by which we will multiply the TD (n) returns.
            if i > terminal_state_indices[index] and index < len(terminal_state_indices) - 1:
                index += 1
                next_terminal_state_index = terminal_state_indices[index]
            td_n_returns = []  # List containing the TD (n) returns.
            xyz = 0
            for n in range(next_terminal_state_index, next_terminal_state_index + 1):
                while i != n:  # Execute until you encounter a terminal state.

                    # Equation to calculate the Monte-Carlo return.
                    td_n_return += self.discount_factor ** t * rewards[i]

                    td_n_returns.append(td_n_return + self.discount_factor ** (t + 1) * next_state_values[i])
                    i += 1  # Go to the next sample.
                    t += 1  # Increasing the exponent by which the discount factor is raised.

                    # Condition to check whether we have reached the end of the replay memory without the episode being
                    # terminated, and if so break. (This can happen with the samples at the end of the replay memory as
                    # we only store the samples till we reach the replay memory size and not till we exceed it with the
                    # episode being terminated.)
                    if i == len(self.offline_replay_memory):
                        # Resetting the Monte-Carlo return value and the exponent to 0.
                        td_n_return = 0
                        t = 0

                        break  # Break from the loop.

                # If for one of the samples towards the end we reach the end of the replay memory and it hasn't
                # terminated, we will go back to the beginning of the for loop to calculate the Monte-Carlo return for
                # the future samples if any for whom the episode hasn't terminated.
                if i == len(self.offline_replay_memory):
                    continue

                # Equation to calculate the Monte-Carlo return.
                td_n_return += self.discount_factor ** t * rewards[i]
                td_n_return += self.discount_factor ** (t + 1) * next_state_values[i]
                # Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of
                # the replay memory.
                td_n_returns.append(td_n_return)

                # Resetting the Monte-Carlo return value and the exponent to 0.
                td_n_return = 0
                t = 0
            if i > terminal_state_indices[index] and index == len(terminal_state_indices) - 1:
                xyz = len(self.offline_replay_memory) - next_terminal_state_index - 1
            values_to_multiply = [td_lambda_value ** x for x in range(next_terminal_state_index + 1 - j + xyz)]
            td_lambda_returns.append((1 - td_lambda_value) * np.dot(values_to_multiply, td_n_returns))

        # Normalizing the returns.
        td_lambda_returns = np.array(td_lambda_returns)
        td_lambda_returns = (td_lambda_returns - np.mean(td_lambda_returns)) / (np.std(td_lambda_returns)
                                                                                + 1e-08)
        td_lambda_returns = td_lambda_returns.tolist()

        return td_lambda_returns

    def replay(self):
        """This is the replay method, that is used to fit the actor and critic networks and synchronize the weights
            between the actor and policy networks."""

        states = [item[0] for item in self.offline_replay_memory]
        states = np.asarray(states).reshape((-1, self.time_period, self.environment.observation_space.n))

        actions = [tf.keras.utils.to_categorical(item[1], self.environment.action_space.n).tolist()
                   for item in self.offline_replay_memory]

        monte_carlo_returns = self.monte_carlo_returns()

        critic_values = self.critic_model.predict(states).flatten()

        # exponential_advantages = [np.exp(1/self.beta * (monte_carlo_returns[i] - critic_values[i]))
        #               for i in range(len(self.offline_replay_memory))]

        advantages = [monte_carlo_returns[i] - critic_values[i]
                      for i in range(len(self.offline_replay_memory))]

        # advantages = [monte_carlo_returns[i] - critic_values[i]
        #               for i in range(len(states))]

        # Fitting the actor model.
        self.actor_model.fit([states, np.asarray(advantages)], np.asarray(actions),
                             batch_size=256, epochs=5, verbose=0)

        # Syncing the weights between the actor and policy models.
        self.policy_model.set_weights(self.actor_model.get_weights())

        # Fitting the critic model.
        self.critic_model.fit(states, np.asarray(monte_carlo_returns), batch_size=256, epochs=5, verbose=0)

    def train(self):
        """This method performs the agent training."""

        average_reward_per_episode_per_iteration = []
        cumulative_average_rewards_per_episode_per_iteration = []

        for iteration in range(self.iterations):
            start = time.time()
            print(f'\n\n Iteration {iteration + 1}')

            # self.offline_replay_memory = []  # Resetting the offline replay memory to be empty.
            total_reward_iteration = 0  # Total reward acquired in this iteration.
            episodes = 0  # Initializing the number of episodes in this iteration to be 0.

            for _ in range(100):
            # while len(self.offline_replay_memory) < self.offline_replay_memory_size:

                # Resetting the environment and starting from a random position.
                state = self.environment.reset()
                state = [state for _ in range(self.time_period)]
                state = deque(state, maxlen=self.time_period)

                done = False  # Initializing the done parameter which indicates whether the environment has terminated
                # or not to False.
                episodes += 1  # Increasing the number of episodes in this iteration.

                while not done:
                    # Selecting an action according to the predicted action probabilities.
                    action_probabilities = (self.policy_model.predict(np.asarray(state).reshape(
                        (-1, self.time_period, self.environment.observation_space.n)))[0])

                    # Adding noise to do exploration and making only legal actions available in a state.
                    # print('\nPrevious Action:', self.environment.previous_action)
                    # print('Current Action:', self.environment.current_action)
                    # print('Action History:', self.environment.action_history, len(self.environment.action_history))
                    # if self.environment.action_history[-16:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                    #     print('omg')
                    #     print('\nPrevious Action:', self.environment.previous_action)
                    #     print('Current Action:', self.environment.current_action)
                    #     print('Action History:', self.environment.action_history, len(self.environment.action_history))
                    #     print('Original Probabilities:', action_probabilities)
                    #     print('Allowed Actions:', self.environment.allowed_actions)
                    #     print('Required Actions:', self.environment.required_actions)
                    #     print('Allowed Actions One Hot:', np.asarray(self.environment.allowed_actions_numbers))

                    # print('Original Probabilities:', action_probabilities)
                    action_probabilities = np.asarray(action_probabilities)
                    allowed_actions_numbers = np.asarray(self.environment.allowed_actions_numbers)
                    allowed_action_probabilities = allowed_actions_numbers * action_probabilities
                    remainder_probability = 1 - np.sum(allowed_action_probabilities)
                    allowed_action_probabilities += \
                        (remainder_probability * allowed_actions_numbers
                         / np.count_nonzero(allowed_actions_numbers))
                    # print('Allowed Actions:', self.environment.allowed_actions)
                    # print('Required Actions:', self.environment.required_actions)
                    # print('Allowed Actions One Hot:', allowed_actions_numbers)
                    # print('Before Noise Allowed Action Probabilities:', allowed_action_probabilities)

                    """Adding noise only to allowed actions:"""
                    allowed_action_probabilities += ((1 / self.environment.action_space.n * (iteration + 1))
                                                     * allowed_actions_numbers)
                    # allowed_action_probabilities += (0.025 * allowed_actions_numbers)
                    allowed_action_probabilities /= np.sum(allowed_action_probabilities)
                    # print('After Noise Allowed Action Probabilities:', allowed_action_probabilities)

                    # DEBUGGING TEST
                    # if len(self.environment.action_history) > 0:
                    #     if self.environment.action_history[-1] == 0:
                    #         print(self.environment.action_history)
                    #         print('Action 0 probabilities:', self.environment.timestep, allowed_action_probabilities)

                    """Adding noise to all actions:"""
                    # action_probabilities += 1 / self.environment.action_space.n * (iteration + 1)
                    # allowed_action_probabilities += 0.025
                    # allowed_action_probabilities /= np.sum(allowed_action_probabilities)

                    # action_probabilities += 0.025
                    # action_probabilities /= np.sum(action_probabilities)
                    # print('New Probabilities:', allowed_action_probabilities)

                    action = np.random.choice(self.environment.action_space.n, p=allowed_action_probabilities)

                    # Taking an action.
                    next_state, reward, done, info = self.environment.step(action)

                    # Incrementing the total reward.
                    total_reward_iteration += reward

                    # Appending the state, action, reward, next state and done to the replay memory.
                    self.offline_replay_memory.append([state, action, reward, next_state, done])

                    # state = next_state  # Setting the current state to be equal to the next state.
                    state.append(next_state)

                    # # This condition ensures that we don't append more values than the size of the replay memory.
                    # if len(self.offline_replay_memory) == self.offline_replay_memory_size:
                    #     break

            # Calculating the average reward per episode for this iteration.
            average_reward_per_episode = total_reward_iteration / episodes
            average_reward_per_episode_per_iteration.append(average_reward_per_episode)

            # Appending the cumulative reward.
            if len(cumulative_average_rewards_per_episode_per_iteration) == 0:
                cumulative_average_rewards_per_episode_per_iteration.append(average_reward_per_episode)
            else:
                cumulative_average_rewards_per_episode_per_iteration.append(
                    average_reward_per_episode + cumulative_average_rewards_per_episode_per_iteration[iteration - 1])

            print('Time to generate samples:', time.time() - start)
            print('Length of Replay Memory:', len(self.offline_replay_memory))

            # Calling the replay method.
            start = time.time()
            self.replay()
            print('Time to train:', time.time() - start)

            self.render_actions(iteration + 1)

        # Calling the plots method to plot the reward dynamics.
        # self.plots(average_reward_per_episode_per_iteration, cumulative_average_rewards_per_episode_per_iteration,
        #            iterations=True)

    def evaluate(self):
        """This method evaluates the performance of the agent after it has finished training."""

        total_steps = 0  # Initializing the total steps taken and total penalties incurred
        # across all episodes.
        episodes = 100  # Number of episodes for which we are going to test the agent's performance.
        rewards_per_episode = []  # Sum of immediate rewards during the episode.
        # gold = 0  # Counter to keep track of the episodes in which the agent reaches the Gold.

        for episode in range(episodes):
            state = self.environment.reset()  # Resetting the environment for every new episode.
            steps = 0  # Initializing the steps taken, and penalties incurred in this episode.
            done = False  # Initializing the done parameter indicating the episode termination to be False.
            total_reward_episode = 0  # Initializing the total reward acquired in this episode to be 0.

            while not done:
                # Always choosing the greedy action.
                action = np.argmax(self.policy_model.predict(
                    np.asarray(state).reshape(-1, self.environment.observation_space.n))[0])

                # Taking the greedy action.
                next_state, reward, done, info = self.environment.step(action)

                total_reward_episode += reward  # Adding the reward acquired on this step to the total reward acquired
                # during the episode.

                state = next_state  # Setting the current state to the next state.

                steps += 1  # Increasing the number of steps taken in this episode.

            rewards_per_episode.append(total_reward_episode)  # Appending the reward acquired during the episode.

            # Appending the cumulative reward.
            if len(self.cumulative_rewards_evaluation) == 0:
                self.cumulative_rewards_evaluation.append(total_reward_episode)
            else:
                self.cumulative_rewards_evaluation.append(
                    total_reward_episode + self.cumulative_rewards_evaluation[episode - 1])

            total_steps += steps  # Adding the steps taken in this episode to the total steps taken across all episodes

        # Printing some statistics after the evaluation of agent's performance is completed.
        print(f"\nEvaluation of agent's performance across {episodes} episodes:")
        print(f"Average number of steps taken per episode: {total_steps / episodes}\n")

        # Calling the plots method to plot the reward dynamics.
        # self.plots(rewards_per_episode, self.cumulative_rewards_evaluation)

    def render_actions(self, iteration):
        # Rendering the actions taken by the agent after learning.
        state = self.environment.reset()  # Resetting the environment for a new episode.
        state = [state for _ in range(self.time_period)]
        state = deque(state, maxlen=self.time_period)
        done = False  # Initializing the done parameter indicating the episode termination to be False.

        # Lists for plotting:
        self.number_of_unvaccinated_susceptible_individuals_list = []
        self.number_of_fully_vaccinated_susceptible_individuals_list = []
        self.number_of_booster_vaccinated_susceptible_individuals_list = []
        self.number_of_susceptible_individuals_list = []

        self.number_of_unvaccinated_exposed_individuals_list = []
        self.number_of_fully_vaccinated_exposed_individuals_list = []
        self.number_of_booster_vaccinated_exposed_individuals_list = []
        self.number_of_exposed_individuals_list = []

        self.number_of_unvaccinated_infected_individuals_list = []
        self.number_of_fully_vaccinated_infected_individuals_list = []
        self.number_of_booster_vaccinated_infected_individuals_list = []
        self.number_of_infected_individuals_list = []

        self.number_of_unvaccinated_hospitalized_individuals_list = []
        self.number_of_fully_vaccinated_hospitalized_individuals_list = []
        self.number_of_booster_vaccinated_hospitalized_individuals_list = []
        self.number_of_hospitalized_individuals_list = []

        self.number_of_unvaccinated_recovered_individuals_list = []
        self.number_of_fully_vaccinated_recovered_individuals_list = []
        self.number_of_booster_vaccinated_recovered_individuals_list = []
        self.number_of_recovered_individuals_list = []

        self.number_of_unvaccinated_deceased_individuals_list = []
        self.number_of_fully_vaccinated_deceased_individuals_list = []
        self.number_of_booster_vaccinated_deceased_individuals_list = []
        self.number_of_deceased_individuals_list = []

        self.number_of_unvaccinated_individuals_list = []
        self.number_of_fully_vaccinated_individuals_list = []
        self.number_of_booster_vaccinated_individuals_list = []

        self.new_cases_list = []

        self.economic_and_social_rate_list = []
        self.action_history = []

        while not done:
            # Always choosing the greedy action.
            # action = np.argmax(self.policy_model.predict(
            #     np.asarray(state).reshape((-1, self.time_period, self.environment.observation_space.n)))[0])

            action_probabilities = (self.policy_model.predict(np.asarray(state).reshape(
                (-1, self.time_period, self.environment.observation_space.n)))[0])

            # print('Original Probabilities:', action_probabilities)
            action_probabilities = np.asarray(action_probabilities)
            allowed_actions_numbers = np.asarray(self.environment.allowed_actions_numbers)
            allowed_action_probabilities = allowed_actions_numbers * action_probabilities
            remainder_probability = 1 - np.sum(allowed_action_probabilities)
            allowed_action_probabilities += \
                (remainder_probability * allowed_actions_numbers
                 / np.count_nonzero(allowed_actions_numbers))

            # print('\nPrevious Action:', self.environment.previous_action)
            # print('Current Action:', self.environment.current_action)
            # print('Action History:', self.environment.action_history, len(self.environment.action_history))
            # print('Original Probabilities:', action_probabilities)
            # print('Allowed Actions:', self.environment.allowed_actions)
            # print('Required Actions:', self.environment.required_actions)
            # print('Allowed Actions One Hot:', np.asarray(self.environment.allowed_actions_numbers))
            # print('Before Noise Allowed Action Probabilities:', allowed_action_probabilities)

            # # allowed_action_probabilities += ((1 / self.environment.action_space.n * iteration)
            # #                                  * self.environment.allowed_actions_numbers)
            # allowed_action_probabilities += (0.025 * allowed_actions_numbers)
            # allowed_action_probabilities /= np.sum(allowed_action_probabilities)

            action = np.argmax(allowed_action_probabilities)

            self.action_history.append(action)

            # Taking the greedy action.
            next_state, reward, done, info = self.environment.step(action)

            # Appending the population statistics to their lists for plotting the graph.
            self.number_of_unvaccinated_susceptible_individuals_list.append(
                self.environment.number_of_unvaccinated_susceptible_individuals)
            self.number_of_fully_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_fully_vaccinated_susceptible_individuals)
            self.number_of_booster_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_booster_vaccinated_susceptible_individuals)
            self.number_of_susceptible_individuals_list.append(self.environment.number_of_susceptible_individuals)

            self.number_of_unvaccinated_exposed_individuals_list.append(
                self.environment.number_of_unvaccinated_exposed_individuals)
            self.number_of_fully_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_fully_vaccinated_exposed_individuals)
            self.number_of_booster_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_booster_vaccinated_exposed_individuals)
            self.number_of_exposed_individuals_list.append(self.environment.number_of_exposed_individuals)

            self.number_of_unvaccinated_infected_individuals_list.append(
                self.environment.number_of_unvaccinated_infected_individuals)
            self.number_of_fully_vaccinated_infected_individuals_list.append(
                self.environment.number_of_fully_vaccinated_infected_individuals)
            self.number_of_booster_vaccinated_infected_individuals_list.append(
                self.environment.number_of_booster_vaccinated_infected_individuals)
            self.number_of_infected_individuals_list.append(self.environment.number_of_infected_individuals)

            self.number_of_unvaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_unvaccinated_hospitalized_individuals)
            self.number_of_fully_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_fully_vaccinated_hospitalized_individuals)
            self.number_of_booster_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_booster_vaccinated_hospitalized_individuals)
            self.number_of_hospitalized_individuals_list.append(self.environment.number_of_hospitalized_individuals)

            self.number_of_unvaccinated_recovered_individuals_list.append(
                self.environment.number_of_unvaccinated_recovered_individuals)
            self.number_of_fully_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_fully_vaccinated_recovered_individuals)
            self.number_of_booster_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_booster_vaccinated_recovered_individuals)
            self.number_of_recovered_individuals_list.append(self.environment.number_of_recovered_individuals)

            self.number_of_unvaccinated_deceased_individuals_list.append(
                self.environment.number_of_unvaccinated_deceased_individuals)
            self.number_of_fully_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_fully_vaccinated_deceased_individuals)
            self.number_of_booster_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_booster_vaccinated_deceased_individuals)
            self.number_of_deceased_individuals_list.append(self.environment.number_of_deceased_individuals)

            self.number_of_unvaccinated_individuals_list.append(
                self.environment.number_of_unvaccinated_individuals)
            self.number_of_fully_vaccinated_individuals_list.append(
                self.environment.number_of_fully_vaccinated_individuals)
            self.number_of_booster_vaccinated_individuals_list.append(
                self.environment.number_of_booster_vaccinated_individuals)

            self.economic_and_social_rate_list.append(self.environment.economic_and_social_rate)

            # state = next_state  # Setting the current state to the next state.
            state.append(next_state)

        # Create a dataframe from lists
        df = pd.DataFrame(
            list(zip(self.environment.covid_data['date'][214:],
                     self.number_of_unvaccinated_individuals_list, self.number_of_fully_vaccinated_individuals_list,
                     self.number_of_booster_vaccinated_individuals_list,
                     self.environment.new_cases,
                     self.number_of_susceptible_individuals_list,
                     self.number_of_exposed_individuals_list, self.number_of_infected_individuals_list,
                     self.number_of_hospitalized_individuals_list, self.number_of_recovered_individuals_list,
                     self.number_of_deceased_individuals_list, self.number_of_unvaccinated_susceptible_individuals_list,
                     self.number_of_fully_vaccinated_susceptible_individuals_list,
                     self.number_of_booster_vaccinated_susceptible_individuals_list,
                     self.number_of_unvaccinated_exposed_individuals_list,
                     self.number_of_fully_vaccinated_exposed_individuals_list,
                     self.number_of_booster_vaccinated_exposed_individuals_list,
                     self.number_of_unvaccinated_infected_individuals_list,
                     self.number_of_fully_vaccinated_infected_individuals_list,
                     self.number_of_booster_vaccinated_infected_individuals_list,
                     self.number_of_unvaccinated_hospitalized_individuals_list,
                     self.number_of_fully_vaccinated_hospitalized_individuals_list,
                     self.number_of_booster_vaccinated_hospitalized_individuals_list,
                     self.number_of_unvaccinated_recovered_individuals_list,
                     self.number_of_fully_vaccinated_recovered_individuals_list,
                     self.number_of_booster_vaccinated_recovered_individuals_list,
                     self.number_of_unvaccinated_deceased_individuals_list,
                     self.number_of_fully_vaccinated_deceased_individuals_list,
                     self.number_of_booster_vaccinated_deceased_individuals_list,
                     self.economic_and_social_rate_list,
                     self.action_history)),
            columns=['date', 'unvaccinated_individuals', 'fully_vaccinated_individuals',
                     'booster_vaccinated_individuals', 'New Cases',
                     'Susceptible', 'Exposed', 'Infected', 'Hospitalized', 'Recovered', 'Deceased',
                     'Susceptible_UV', 'Susceptible_FV', 'Susceptible_BV', 'Exposed_UV', 'Exposed_FV', 'Exposed_BV',
                     'Infected_UV', 'Infected_FV', 'Infected_BV',
                     'Hospitalized_UV', 'Hospitalized_FV', 'Hospitalized_BV',
                     'Recovered_UV', 'Recovered_FV', 'Recovered_BV', 'Deceased_UV',	'Deceased_FV', 'Deceased_BV',
                     'Economic and Social Perception Rate', 'Action'])

        df.to_csv(f'./Results/DataFrames/{self.date_and_time}/{iteration}.csv')

        print('Timestep:', self.environment.timestep,
              'Number of Susceptible People:', self.environment.number_of_susceptible_individuals,
              'Number of Exposed People:', self.environment.number_of_exposed_individuals,
              'Number of Infected People:', self.environment.number_of_infected_individuals,
              'Number of Hospitalized People:', self.environment.number_of_hospitalized_individuals,
              'Number of Recovered People:', self.environment.number_of_recovered_individuals,
              'Number of Deceased People:', self.environment.number_of_deceased_individuals,
              'GDP:', self.environment.economic_and_social_rate)

        self.population_dynamics[iteration] = \
            [self.number_of_susceptible_individuals_list, self.number_of_exposed_individuals_list,
             self.number_of_infected_individuals_list, self.number_of_hospitalized_individuals_list,
             self.number_of_recovered_individuals_list, self.number_of_deceased_individuals_list,
             self.economic_and_social_rate_list]

        print(len(self.population_dynamics[iteration]), len(self.population_dynamics[iteration][0]))
        print('population Dynamics:', self.population_dynamics[iteration])
        print('Day 30 Susceptible:', self.population_dynamics[iteration][0][29],
              'Day 30 Exposed:', self.population_dynamics[iteration][1][29],
              'Day 30 Infected:', self.population_dynamics[iteration][2][29],
              'Day 30 Hospitalized:', self.population_dynamics[iteration][3][29],
              'Day 30 Recovered:', self.population_dynamics[iteration][4][29],
              'Day 30 Deceased:', self.population_dynamics[iteration][5][29],
              'Day 30 ESR:', self.population_dynamics[iteration][6][29])

        print('Day 60 Susceptible:', self.population_dynamics[iteration][0][59],
              'Day 60 Exposed:', self.population_dynamics[iteration][1][59],
              'Day 60 Infected:', self.population_dynamics[iteration][2][59],
              'Day 60 Hospitalized:', self.population_dynamics[iteration][3][59],
              'Day 60 Recovered:', self.population_dynamics[iteration][4][59],
              'Day 60 Deceased:', self.population_dynamics[iteration][5][59],
              'Day 60 ESR:', self.population_dynamics[iteration][6][59])

        print('Day 90 Susceptible:', self.population_dynamics[iteration][0][89],
              'Day 90 Exposed:', self.population_dynamics[iteration][1][89],
              'Day 90 Infected:', self.population_dynamics[iteration][2][89],
              'Day 90 Hospitalized:', self.population_dynamics[iteration][3][89],
              'Day 90 Recovered:', self.population_dynamics[iteration][4][89],
              'Day 90 Deceased:', self.population_dynamics[iteration][5][89],
              'Day 90 ESR:', self.population_dynamics[iteration][6][89])

        print('Day 120 Susceptible:', self.population_dynamics[iteration][0][119],
              'Day 120 Exposed:', self.population_dynamics[iteration][1][119],
              'Day 120 Infected:', self.population_dynamics[iteration][2][119],
              'Day 120 Hospitalized:', self.population_dynamics[iteration][3][119],
              'Day 120 Recovered:', self.population_dynamics[iteration][4][119],
              'Day 120 Deceased:', self.population_dynamics[iteration][5][119],
              'Day 120 ESR:', self.population_dynamics[iteration][6][119])

        print('Day 180 Susceptible:', self.population_dynamics[iteration][0][179],
              'Day 180 Exposed:', self.population_dynamics[iteration][1][179],
              'Day 180 Infected:', self.population_dynamics[iteration][2][179],
              'Day 180 Hospitalized:', self.population_dynamics[iteration][3][179],
              'Day 180 Recovered:', self.population_dynamics[iteration][4][179],
              'Day 180 Deceased:', self.population_dynamics[iteration][5][179],
              'Day 180 ESR:', self.population_dynamics[iteration][6][179])

        print('Action History:', self.action_history)

        # self.environment.render()  # Rendering the environment.

    @staticmethod
    def plots(rewards_per_episode, cumulative_rewards, iterations=False):
        """This method plots the reward dynamics and epsilon decay.

        :param iterations: Boolean indicating that we are plotting for iterations and not episodes.
        :param rewards_per_episode: List containing the reward values per episode.
        :param cumulative_rewards: List containing the cumulative reward values per episode."""

        plt.figure(figsize=(20, 10))
        plt.plot(rewards_per_episode, 'ro')
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward Per Episode')
            plt.title('Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.title('Rewards Per Episode (During Evaluation)')
        plt.grid()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(cumulative_rewards)
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Cumulative Average Reward Per Episode')
            plt.title('Cumulative Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Cumulative Reward Per Episode')
            plt.title('Cumulative Rewards Per Episode (During Evaluation)')
        plt.grid()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()


# Instantiating the deterministic and stochastic Wumpus World environment.
disease_mitigation_environment = DiseaseMitigation(state_name='epidemiological_model_data',
                                                   state_population=19_453_734,
                                                   start_date='11/01/2021')

awr_agent = AdvantageWeightedRegression(disease_mitigation_environment, alternate_network=False,
                                        offline_memory_size=100_000, iterations=10)


# # Actual Data
# covid_data = pd.read_csv('./covid_ny.csv')
# active_cases = [covid_data['Active Cases'][i] for i in range(203, 383)]
# total_recoveries = [covid_data['Total Recovered'][i] for i in range(203, 383)]
# total_deaths = [covid_data['Total Deaths'][i] for i in range(203, 383)]
# healthy_individuals = [19_453_561 - covid_data['Active Cases'][i] - covid_data['Total Recovered'][i]
#                        - covid_data['Total Deaths'][i] for i in range(203, 383)]
#
#
# # Comparison of Infected Individuals
# plt.figure(figsize=(15, 10))
# plt.plot(awr_agent.population_dynamics[1][0], label='AWR', color='green', linewidth=7)
# plt.plot(active_cases, label='NY Data', linestyle='dashed', color='red', linewidth=7)
# plt.legend(fontsize=28)
# plt.xlabel('Days', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title('Dynamics of Infected Individuals', fontsize=36)
# plt.grid()
# plt.xticks(np.arange(0, 181, 30), fontsize=30)
# plt.yticks([200000, 400000, 600000, 800000], ["200K", "400K", "600K", "800K"], fontsize=30)
# plt.xlim(xmin=0, xmax=180)
# plt.ylim(ymin=0)
# plt.show()
#
#
# # Comparison of Dead Individuals
# plt.figure(figsize=(15, 10))
# plt.plot(awr_agent.population_dynamics[1][2], label='AWR', color='green', linewidth=7)
# plt.plot(total_deaths, label='NY Data', linestyle='dashed', color='red', linewidth=7)
# plt.legend(fontsize=28)
# plt.xlabel('Days', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title('Dynamics of Removed Individuals', fontsize=36)
# plt.grid()
# plt.xticks(np.arange(0, 181, 30), fontsize=30)
# plt.yticks([35000, 40000, 45000, 50000], ["35K", "40K", "45K", "50K"], fontsize=30)
# plt.xlim(xmin=0, xmax=180)
# plt.show()
#
#
# # Comparison of Healthy Individuals
# plt.figure(figsize=(15, 10))
# # Infected Comparison
# plt.plot(awr_agent.population_dynamics[1][3], label='AWR', color='green', linewidth=7)
# plt.plot(healthy_individuals, label='NY Data', linestyle='dashed', color='red', linewidth=7)
# plt.legend(fontsize=28)
# plt.xlabel('Days', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title('Dynamics of Healthy Individuals', fontsize=36)
# plt.grid()
# plt.xticks(np.arange(0, 181, 30), fontsize=30)
# plt.yticks(fontsize=30)
# plt.yticks([17_500_000, 18_000_000, 18_500_000, 19_000_000], ["17.5M", "18M", "18.5M", "19M"], fontsize=30)
# plt.xlim(xmin=0, xmax=180)
# # plt.ylim(ymin=0)
# plt.show()
