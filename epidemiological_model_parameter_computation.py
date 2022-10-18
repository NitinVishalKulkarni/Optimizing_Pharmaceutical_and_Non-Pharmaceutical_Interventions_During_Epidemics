# Imports
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from time import time


class ParameterComputation:
    """This class computes the parameters for the epidemiological model."""

    def __init__(self, filepath='./epidemiological_model_data.csv', population=19_453_734, compartment_names=None,
                 data_split=None, constrained_beta=True):
        """This method initializes the parameters.

        :parameter filepath: String - Filepath of the epidemiological model data.
        :parameter population: Integer - Population of the epidemic region.
        :parameter compartment_names: Array - Array of strings specifying the compartments in the epidemiological model.
        :parameter data_split: Integer - Integer specifying the time period of days in which we will split the data.
        :parameter constrained_beta - Boolean specifying whether the exposure rate beta is unconstrained."""

        self.epidemiological_data = pd.read_csv(filepath)
        self.population = population
        self.parameters = self.parameter_initialization(constrained_beta)

        self.compartment_names = compartment_names
        self.data_split = data_split

        if self.data_split:

            self.epidemiological_data_splits = [
                self.epidemiological_data[t * data_split: min((t + 1) * data_split, len(self.epidemiological_data))]
                for t in range(int(np.ceil(len(self.epidemiological_data) / self.data_split)))]
            print('Length of Epidemiological Data Splits:', len(self.epidemiological_data_splits))
            print('Shape of the first split:', self.epidemiological_data_splits[0].shape)
            print('Shape of the last split:', self.epidemiological_data_splits[-1].shape)

            self.epidemiological_compartment_values_data_splits = \
                [self.epidemiological_data_splits[i][self.compartment_names].values
                 for i in range(len(self.epidemiological_data_splits))]
            print('\n\nLength of Epidemiological Compartment Values Data Splits:',
                  len(self.epidemiological_compartment_values_data_splits))
            print('Shape of the first split:', self.epidemiological_compartment_values_data_splits[0].shape)
            print('Shape of the last split:', self.epidemiological_compartment_values_data_splits[-1].shape)

            self.t_data_splits = \
                [np.linspace(0, self.epidemiological_data_splits[i].shape[0] -
                             1, self.epidemiological_data_splits[i].shape[0])
                 for i in range(len(self.epidemiological_data_splits))]
            print('\n\nLength of T Data Splits:',
                  len(self.t_data_splits))
            print('Shape of the first split:', self.t_data_splits[0].shape)
            print('Shape of the last split:', self.t_data_splits[-1].shape)

            self.y0_data_splits = [[self.epidemiological_data_splits[j][f'{self.compartment_names[i]}'].iloc[0]
                                    for i in range(len(self.compartment_names))]
                                   for j in range(len(self.epidemiological_data_splits))]
            print('\n\nLength of y0 Data Splits:', len(self.y0_data_splits))
            print('Length of the first split:', len(self.y0_data_splits[0]))
            print('Length of the last split:', len(self.y0_data_splits[-1]))

        self.epidemiological_compartment_values = self.epidemiological_data[self.compartment_names].values
        self.t = np.linspace(0, self.epidemiological_data.shape[0] - 1, self.epidemiological_data.shape[0])

        # Initial values of population dynamics.
        self.y0 = [self.epidemiological_data[f'{self.compartment_names[i]}'].iloc[0]
                   for i in range(len(self.compartment_names))]

        self.parameter_list_data_splits = []
        self.counter = 1

    @staticmethod
    def parameter_initialization(constrained_beta=True):
        """This method initializes the parameter values for the epidemiological model.

        :return parameters: Parameters for the epidemiological model."""

        # Set parameters including bounds
        parameters = Parameters()

        # Exposure rate.
        if constrained_beta:
            parameters.add('beta_i', value=0.37, min=0.0, max=1)
        else:
            parameters.add('beta_i', value=0.37, min=-np.inf, max=1)

        # Population mixing coefficient.
        parameters.add('alpha', value=1, min=0, max=1)

        # Infection rates for exposed individuals.
        parameters.add('zeta_uv', value=0.615, min=0, max=0.4)
        parameters.add('zeta_pv', value=0.039, min=0, max=0.4)
        parameters.add('zeta_fv', value=0.346, min=0, max=0.4)

        # Hospitalization rates for infected individuals.
        parameters.add('delta_uv', value=0.715, min=0, max=0.1)
        parameters.add('delta_pv', value=0.05, min=0, max=0.1)
        parameters.add('delta_fv', value=0.235, min=0, max=0.1)

        # Recovery rates for infected individuals.
        parameters.add('gamma_i_uv', value=0.6108, min=0.0, max=1)
        parameters.add('gamma_i_pv', value=0.0388, min=0.0, max=1)
        parameters.add('gamma_i_fv', value=0.3502, min=0.0, max=1)

        # Recovery rates for hospitalized individuals.
        parameters.add('gamma_h_uv', value=0.6108, min=0.0, max=1)
        parameters.add('gamma_h_pv', value=0.0388, min=0.0, max=1)
        parameters.add('gamma_h_fv', value=0.3502, min=0.0, max=1)

        # Death rates for infected individuals.
        parameters.add('mu_i_uv', value=0.000111942027, min=0.0, max=0.1)
        parameters.add('mu_i_pv', value=0.000111942027, min=0.0, max=0.1)
        parameters.add('mu_i_fv', value=0.000111942027, min=0.0, max=0.1)

        # Death rates for hospitalized individuals.
        parameters.add('mu_h_uv', value=0.000111942027, min=0.0, max=0.2)
        parameters.add('mu_h_pv', value=0.000111942027, min=0.0, max=0.2)
        parameters.add('mu_h_fv', value=0.000111942027, min=0.0, max=0.2)

        # Rate at which previously exposed individuals become susceptible again.
        parameters.add('exp_to_suv', value=0.000111942027, min=0.0, max=1)
        parameters.add('exp_to_spv', value=0.000111942027, min=0.0, max=1)
        parameters.add('exp_to_sfv', value=0.000111942027, min=0.0, max=1)

        return parameters

    # noinspection DuplicatedCode
    def differential_equations(self, y, t, population, parameters, call_signature_ode_int=True,
                               differential_equations_version=1):
        """This method models the differential equations for the epidemiological model.

        :param y: Vector of sub-compartment population dynamics
        :param t: Time span of simulation
        :param population: Total Population
        :param parameters: Parameter values
        :param call_signature_ode_int: Boolean - Indicates if the calling signature of the differential_equations method
                                                 is that of scipy's odeint method.
        :param differential_equations_version: Integer: Version of differential equations/model to be used.

        :returns derivatives of the model compartments."""

        if not call_signature_ode_int:
            t, y = y, t

        # VERSION 1:
        if differential_equations_version == 1:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
               We allow recovered individuals to be susceptible again. Model accounts for vaccination rates. We model 
               susceptible, exposed, infected, and recovered people to be vaccinated. Exposure rate is independently
               computed."""

            # Sub-compartments
            s_uv, s_pv, s_fv, e_uv, e_pv, e_fv, i_uv, i_pv, i_fv, \
                h_uv, h_pv, h_fv, r_uv, r_pv, r_fv, d_uv, d_pv, d_fv = y

            # Force of infection
            total_infections = max((i_uv + i_pv + i_fv), 1)

            if 'rec_to_suv' not in parameters:
                parameters.add('rec_to_suv', value=0.111942027, min=0.0, max=1)
                parameters.add('rec_to_spv', value=0.111942027, min=0.0, max=1)
                parameters.add('rec_to_sfv', value=0.111942027, min=0.0, max=1)

            # Parameter Values
            beta_i = parameters['beta_i'].value
            alpha = parameters['alpha'].value

            zeta_uv = parameters['zeta_uv'].value
            zeta_pv = parameters['zeta_pv'].value
            zeta_fv = parameters['zeta_fv'].value

            delta_uv = parameters['delta_uv'].value
            delta_pv = parameters['delta_pv'].value
            delta_fv = parameters['delta_fv'].value

            mu_i_uv = parameters['mu_i_uv'].value
            mu_i_pv = parameters['mu_i_pv'].value
            mu_i_fv = parameters['mu_i_fv'].value

            mu_h_uv = parameters['mu_h_uv'].value
            mu_h_pv = parameters['mu_h_pv'].value
            mu_h_fv = parameters['mu_h_fv'].value

            gamma_i_uv = parameters['gamma_i_uv'].value
            gamma_i_pv = parameters['gamma_i_pv'].value
            gamma_i_fv = parameters['gamma_i_fv'].value

            gamma_h_uv = parameters['gamma_h_uv'].value
            gamma_h_pv = parameters['gamma_h_pv'].value
            gamma_h_fv = parameters['gamma_h_fv'].value

            exp_to_s_uv = parameters['exp_to_suv'].value
            exp_to_s_pv = parameters['exp_to_spv'].value
            exp_to_s_fv = parameters['exp_to_sfv'].value

            rec_to_s_uv = parameters['rec_to_suv'].value
            rec_to_s_pv = parameters['rec_to_spv'].value
            rec_to_s_fv = parameters['rec_to_sfv'].value

            # Ordinary Differential Equations.
            beta = beta_i

            # Susceptible
            ds_uv_dt = -beta * s_uv * (total_infections ** alpha) / population + exp_to_s_uv * e_uv + \
                rec_to_s_uv * r_uv - self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv
            ds_pv_dt = -beta * s_pv * (total_infections ** alpha) / population + exp_to_s_pv * e_pv + \
                rec_to_s_pv * r_pv + self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv
            ds_fv_dt = -beta * s_fv * (total_infections ** alpha) / population + exp_to_s_fv * e_fv + \
                rec_to_s_fv * r_fv + self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv

            # Exposed
            de_uv_dt = beta * s_uv * (total_infections ** alpha) / population - zeta_uv * e_uv - exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * e_uv
            de_pv_dt = beta * s_pv * (total_infections ** alpha) / population - zeta_pv * e_pv - exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * e_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv
            de_fv_dt = beta * s_fv * (total_infections ** alpha) / population - zeta_fv * e_fv - exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * e_pv

            # Infected
            di_uv_dt = zeta_uv * e_uv - delta_uv * i_uv - gamma_i_uv * i_uv - mu_i_uv * i_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_uv
            di_pv_dt = zeta_pv * e_pv - delta_pv * i_pv - gamma_i_pv * i_pv - mu_i_pv * i_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_pv
            di_fv_dt = zeta_fv * e_fv - delta_fv * i_fv - gamma_i_fv * i_fv - mu_i_fv * i_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_pv

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_pv_dt = delta_pv * i_pv - gamma_h_pv * h_pv - mu_h_pv * h_pv
            dh_fv_dt = delta_fv * i_fv - gamma_h_fv * h_fv - mu_h_fv * h_fv

            # Recovered
            dr_uv_dt = gamma_i_uv * i_uv + gamma_h_uv * h_uv - rec_to_s_uv * r_uv +\
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_uv
            dr_pv_dt = gamma_i_pv * i_pv + gamma_h_uv * h_pv - rec_to_s_pv * r_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv
            dr_fv_dt = gamma_i_fv * i_fv + gamma_h_uv * h_fv - rec_to_s_fv * r_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_pv

            # Dead
            dd_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            dd_pv_dt = mu_i_pv * i_pv + mu_h_pv * h_pv
            dd_fv_dt = mu_i_fv * i_fv + mu_h_fv * h_fv

            return ds_uv_dt, ds_pv_dt, ds_fv_dt, de_uv_dt, de_pv_dt, de_fv_dt, di_uv_dt, di_pv_dt, di_fv_dt, \
                dh_uv_dt, dh_pv_dt, dh_fv_dt, dr_uv_dt, dr_pv_dt, dr_fv_dt, dd_uv_dt, dd_pv_dt, dd_fv_dt

        # VERSION 2:
        elif differential_equations_version == 2:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
               We don't allow recovered individuals to be susceptible again. Model accounts for vaccination rates.
               We model susceptible, exposed, infected, and recovered people to be vaccinated. Exposure rate is
               independently computed."""

            # Sub-compartments
            s_uv, s_pv, s_fv, e_uv, e_pv, e_fv, i_uv, i_pv, i_fv, \
                h_uv, h_pv, h_fv, r_uv, r_pv, r_fv, d_uv, d_pv, d_fv = y

            # Force of infection
            total_infections = max((i_uv + i_pv + i_fv), 1)

            # Parameter Values
            beta_i = parameters['beta_i'].value
            alpha = parameters['alpha'].value

            zeta_uv = parameters['zeta_uv'].value
            zeta_pv = parameters['zeta_pv'].value
            zeta_fv = parameters['zeta_fv'].value

            delta_uv = parameters['delta_uv'].value
            delta_pv = parameters['delta_pv'].value
            delta_fv = parameters['delta_fv'].value

            mu_i_uv = parameters['mu_i_uv'].value
            mu_i_pv = parameters['mu_i_pv'].value
            mu_i_fv = parameters['mu_i_fv'].value

            mu_h_uv = parameters['mu_h_uv'].value
            mu_h_pv = parameters['mu_h_pv'].value
            mu_h_fv = parameters['mu_h_fv'].value

            gamma_i_uv = parameters['gamma_i_uv'].value
            gamma_i_pv = parameters['gamma_i_pv'].value
            gamma_i_fv = parameters['gamma_i_fv'].value

            gamma_h_uv = parameters['gamma_h_uv'].value
            gamma_h_pv = parameters['gamma_h_pv'].value
            gamma_h_fv = parameters['gamma_h_fv'].value

            exp_to_s_uv = parameters['exp_to_suv'].value
            exp_to_s_pv = parameters['exp_to_spv'].value
            exp_to_s_fv = parameters['exp_to_sfv'].value

            # Ordinary Differential Equations.
            beta = beta_i

            # Susceptible
            ds_uv_dt = -beta * s_uv * (total_infections ** alpha) / population + exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * s_uv
            ds_pv_dt = -beta * s_pv * (total_infections ** alpha) / population + exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * s_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv
            ds_fv_dt = -beta * s_fv * (total_infections ** alpha) / population + exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * s_pv

            # Exposed
            de_uv_dt = beta * s_uv * (total_infections ** alpha) / population - zeta_uv * e_uv - exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv
            de_pv_dt = beta * s_pv * (total_infections ** alpha) / population - zeta_pv * e_pv - exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv
            de_fv_dt = beta * s_fv * (total_infections ** alpha) / population - zeta_fv * e_fv - exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv

            # Infected
            di_uv_dt = zeta_uv * e_uv - delta_uv * i_uv - gamma_i_uv * i_uv - mu_i_uv * i_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_uv
            di_pv_dt = zeta_pv * e_pv - delta_pv * i_pv - gamma_i_pv * i_pv - mu_i_pv * i_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_pv
            di_fv_dt = zeta_fv * e_fv - delta_fv * i_fv - gamma_i_fv * i_fv - mu_i_fv * i_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * i_pv

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_pv_dt = delta_pv * i_pv - gamma_h_pv * h_pv - mu_h_pv * h_pv
            dh_fv_dt = delta_fv * i_fv - gamma_h_fv * h_fv - mu_h_fv * h_fv

            # Recovered
            dr_uv_dt = gamma_i_uv * i_uv + gamma_h_uv * h_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_uv
            dr_pv_dt = gamma_i_pv * i_pv + gamma_h_uv * h_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv
            dr_fv_dt = gamma_i_fv * i_fv + gamma_h_uv * h_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                    min(int(t), len(self.epidemiological_data) - 1)] * r_pv

            # Dead
            dd_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            dd_pv_dt = mu_i_pv * i_pv + mu_h_pv * h_pv
            dd_fv_dt = mu_i_fv * i_fv + mu_h_fv * h_fv

            return ds_uv_dt, ds_pv_dt, ds_fv_dt, de_uv_dt, de_pv_dt, de_fv_dt, di_uv_dt, di_pv_dt, di_fv_dt, \
                dh_uv_dt, dh_pv_dt, dh_fv_dt, dr_uv_dt, dr_pv_dt, dr_fv_dt, dd_uv_dt, dd_pv_dt, dd_fv_dt

        # VERSION 3:
        elif differential_equations_version == 3:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
               We allow recovered individuals to be susceptible again. Model accounts for vaccination rates. We model 
               susceptible, exposed, infected, and recovered people to be vaccinated. Exposure rate is dependent on the
               number of new cases."""

            # Sub-compartments
            s_uv, s_pv, s_fv, e_uv, e_pv, e_fv, i_uv, i_pv, i_fv, \
                h_uv, h_pv, h_fv, r_uv, r_pv, r_fv, d_uv, d_pv, d_fv = y

            # Force of infection
            total_infections = max((i_uv + i_pv + i_fv), 1)

            if 'rec_to_suv' not in parameters:
                parameters.add('rec_to_suv', value=0.111942027, min=0.0, max=1)
                parameters.add('rec_to_spv', value=0.111942027, min=0.0, max=1)
                parameters.add('rec_to_sfv', value=0.111942027, min=0.0, max=1)

            # Parameter Values
            beta_i = parameters['beta_i'].value
            alpha = parameters['alpha'].value

            zeta_uv = parameters['zeta_uv'].value
            zeta_pv = parameters['zeta_pv'].value
            zeta_fv = parameters['zeta_fv'].value

            delta_uv = parameters['delta_uv'].value
            delta_pv = parameters['delta_pv'].value
            delta_fv = parameters['delta_fv'].value

            mu_i_uv = parameters['mu_i_uv'].value
            mu_i_pv = parameters['mu_i_pv'].value
            mu_i_fv = parameters['mu_i_fv'].value

            mu_h_uv = parameters['mu_h_uv'].value
            mu_h_pv = parameters['mu_h_pv'].value
            mu_h_fv = parameters['mu_h_fv'].value

            gamma_i_uv = parameters['gamma_i_uv'].value
            gamma_i_pv = parameters['gamma_i_pv'].value
            gamma_i_fv = parameters['gamma_i_fv'].value

            gamma_h_uv = parameters['gamma_h_uv'].value
            gamma_h_pv = parameters['gamma_h_pv'].value
            gamma_h_fv = parameters['gamma_h_fv'].value

            exp_to_s_uv = parameters['exp_to_suv'].value
            exp_to_s_pv = parameters['exp_to_spv'].value
            exp_to_s_fv = parameters['exp_to_sfv'].value

            rec_to_s_uv = parameters['rec_to_suv'].value
            rec_to_s_pv = parameters['rec_to_spv'].value
            rec_to_s_fv = parameters['rec_to_sfv'].value

            # Ordinary Differential Equations.
            beta = beta_i * self.epidemiological_data['New Cases'].iloc[min(int(t), len(self.epidemiological_data) - 1)]

            # Susceptible
            ds_uv_dt = -beta * s_uv * (total_infections ** alpha) / population + exp_to_s_uv * e_uv + \
                rec_to_s_uv * r_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv
            ds_pv_dt = -beta * s_pv * (total_infections ** alpha) / population + exp_to_s_pv * e_pv + \
                rec_to_s_pv * r_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv
            ds_fv_dt = -beta * s_fv * (total_infections ** alpha) / population + exp_to_s_fv * e_fv + \
                rec_to_s_fv * r_fv + self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv

            # Exposed
            de_uv_dt = beta * s_uv * (total_infections ** alpha) / population - zeta_uv * e_uv - exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv
            de_pv_dt = beta * s_pv * (total_infections ** alpha) / population - zeta_pv * e_pv - exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv
            de_fv_dt = beta * s_fv * (total_infections ** alpha) / population - zeta_fv * e_fv - exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv

            # Infected
            di_uv_dt = zeta_uv * e_uv - delta_uv * i_uv - gamma_i_uv * i_uv - mu_i_uv * i_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_uv
            di_pv_dt = zeta_pv * e_pv - delta_pv * i_pv - gamma_i_pv * i_pv - mu_i_pv * i_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_pv
            di_fv_dt = zeta_fv * e_fv - delta_fv * i_fv - gamma_i_fv * i_fv - mu_i_fv * i_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * i_pv

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_pv_dt = delta_pv * i_pv - gamma_h_pv * h_pv - mu_h_pv * h_pv
            dh_fv_dt = delta_fv * i_fv - gamma_h_fv * h_fv - mu_h_fv * h_fv

            # Recovered
            dr_uv_dt = gamma_i_uv * i_uv + gamma_h_uv * h_uv - rec_to_s_uv * r_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv
            dr_pv_dt = gamma_i_pv * i_pv + gamma_h_uv * h_pv - rec_to_s_pv * r_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv
            dr_fv_dt = gamma_i_fv * i_fv + gamma_h_uv * h_fv - rec_to_s_fv * r_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv

            # Dead
            dd_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            dd_pv_dt = mu_i_pv * i_pv + mu_h_pv * h_pv
            dd_fv_dt = mu_i_fv * i_fv + mu_h_fv * h_fv

            return ds_uv_dt, ds_pv_dt, ds_fv_dt, de_uv_dt, de_pv_dt, de_fv_dt, di_uv_dt, di_pv_dt, di_fv_dt, \
                dh_uv_dt, dh_pv_dt, dh_fv_dt, dr_uv_dt, dr_pv_dt, dr_fv_dt, dd_uv_dt, dd_pv_dt, dd_fv_dt

        # VERSION 4:
        elif differential_equations_version == 4:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
               We don't allow recovered individuals to be susceptible again. Model accounts for vaccination rates.
               We model susceptible, exposed, and recovered people to be vaccinated. Exposure rate is 
               dependent on the number of new cases."""

            # Sub-compartments
            s_uv, s_pv, s_fv, e_uv, e_pv, e_fv, i_uv, i_pv, i_fv, \
                h_uv, h_pv, h_fv, r_uv, r_pv, r_fv, d_uv, d_pv, d_fv = y

            # Force of infection
            total_infections = max((i_uv + i_pv + i_fv), 1)

            # Parameter Values
            beta_i = parameters['beta_i'].value
            alpha = parameters['alpha'].value

            zeta_uv = parameters['zeta_uv'].value
            zeta_pv = parameters['zeta_pv'].value
            zeta_fv = parameters['zeta_fv'].value

            delta_uv = parameters['delta_uv'].value
            delta_pv = parameters['delta_pv'].value
            delta_fv = parameters['delta_fv'].value

            mu_i_uv = parameters['mu_i_uv'].value
            mu_i_pv = parameters['mu_i_pv'].value
            mu_i_fv = parameters['mu_i_fv'].value

            mu_h_uv = parameters['mu_h_uv'].value
            mu_h_pv = parameters['mu_h_pv'].value
            mu_h_fv = parameters['mu_h_fv'].value

            gamma_i_uv = parameters['gamma_i_uv'].value
            gamma_i_pv = parameters['gamma_i_pv'].value
            gamma_i_fv = parameters['gamma_i_fv'].value

            gamma_h_uv = parameters['gamma_h_uv'].value
            gamma_h_pv = parameters['gamma_h_pv'].value
            gamma_h_fv = parameters['gamma_h_fv'].value

            exp_to_s_uv = parameters['exp_to_suv'].value
            exp_to_s_pv = parameters['exp_to_spv'].value
            exp_to_s_fv = parameters['exp_to_sfv'].value

            # Ordinary Differential Equations.
            beta = beta_i * self.epidemiological_data['New Cases'].iloc[min(int(t), len(self.epidemiological_data) - 1)]

            # Susceptible
            ds_uv_dt = -beta * s_uv * (total_infections ** alpha) / population + exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv
            ds_pv_dt = -beta * s_pv * (total_infections ** alpha) / population + exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv
            ds_fv_dt = -beta * s_fv * (total_infections ** alpha) / population + exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv

            # Exposed
            de_uv_dt = beta * s_uv * (total_infections ** alpha) / population - zeta_uv * e_uv - exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv
            de_pv_dt = beta * s_pv * (total_infections ** alpha) / population - zeta_pv * e_pv - exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv
            de_fv_dt = beta * s_fv * (total_infections ** alpha) / population - zeta_fv * e_fv - exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv

            # Infected
            di_uv_dt = zeta_uv * e_uv - delta_uv * i_uv - gamma_i_uv * i_uv - mu_i_uv * i_uv
            di_pv_dt = zeta_pv * e_pv - delta_pv * i_pv - gamma_i_pv * i_pv - mu_i_pv * i_pv
            di_fv_dt = zeta_fv * e_fv - delta_fv * i_fv - gamma_i_fv * i_fv - mu_i_fv * i_fv

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_pv_dt = delta_pv * i_pv - gamma_h_pv * h_pv - mu_h_pv * h_pv
            dh_fv_dt = delta_fv * i_fv - gamma_h_fv * h_fv - mu_h_fv * h_fv

            # Recovered
            dr_uv_dt = gamma_i_uv * i_uv + gamma_h_uv * h_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv
            dr_pv_dt = gamma_i_pv * i_pv + gamma_h_uv * h_pv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv
            dr_fv_dt = gamma_i_fv * i_fv + gamma_h_uv * h_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv

            # Dead
            dd_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            dd_pv_dt = mu_i_pv * i_pv + mu_h_pv * h_pv
            dd_fv_dt = mu_i_fv * i_fv + mu_h_fv * h_fv

            return ds_uv_dt, ds_pv_dt, ds_fv_dt, de_uv_dt, de_pv_dt, de_fv_dt, di_uv_dt, di_pv_dt, di_fv_dt, \
                dh_uv_dt, dh_pv_dt, dh_fv_dt, dr_uv_dt, dr_pv_dt, dr_fv_dt, dd_uv_dt, dd_pv_dt, dd_fv_dt

        # VERSION 5:
        elif differential_equations_version == 5:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
               We don't allow recovered individuals to be susceptible again. Model accounts for vaccination rates.
               We model susceptible, exposed, and recovered people to be vaccinated. Exposure rate is 
               dependent on the number of new cases."""

            # Sub-compartments
            s_uv, s_pv, s_fv, e_uv, e_pv, e_fv, i_uv, i_pv, i_fv, \
                r_uv, r_pv, r_fv, d_uv, d_pv, d_fv = y

            # Force of infection
            total_infections = max((i_uv + i_pv + i_fv), 1)

            # Parameter Values
            beta_i = parameters['beta_i'].value
            alpha = parameters['alpha'].value

            zeta_uv = parameters['zeta_uv'].value
            zeta_pv = parameters['zeta_pv'].value
            zeta_fv = parameters['zeta_fv'].value

            mu_i_uv = parameters['mu_i_uv'].value
            mu_i_pv = parameters['mu_i_pv'].value
            mu_i_fv = parameters['mu_i_fv'].value

            gamma_i_uv = parameters['gamma_i_uv'].value
            gamma_i_pv = parameters['gamma_i_pv'].value
            gamma_i_fv = parameters['gamma_i_fv'].value

            exp_to_s_uv = parameters['exp_to_suv'].value
            exp_to_s_pv = parameters['exp_to_spv'].value
            exp_to_s_fv = parameters['exp_to_sfv'].value

            # Ordinary Differential Equations.
            beta = beta_i * self.epidemiological_data['New Cases'].iloc[min(int(t), len(self.epidemiological_data) - 1)]

            # Susceptible
            ds_uv_dt = -beta * s_uv * (total_infections ** alpha) / population + exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv
            ds_pv_dt = -beta * s_pv * (total_infections ** alpha) / population + exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv
            ds_fv_dt = -beta * s_fv * (total_infections ** alpha) / population + exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * s_pv

            # Exposed
            de_uv_dt = beta * s_uv * (total_infections ** alpha) / population - zeta_uv * e_uv - exp_to_s_uv * e_uv - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv
            de_pv_dt = beta * s_pv * (total_infections ** alpha) / population - zeta_pv * e_pv - exp_to_s_pv * e_pv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv
            de_fv_dt = beta * s_fv * (total_infections ** alpha) / population - zeta_fv * e_fv - exp_to_s_fv * e_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * e_pv

            # Infected
            di_uv_dt = zeta_uv * e_uv - gamma_i_uv * i_uv - mu_i_uv * i_uv
            di_pv_dt = zeta_pv * e_pv - gamma_i_pv * i_pv - mu_i_pv * i_pv
            di_fv_dt = zeta_fv * e_fv - gamma_i_fv * i_fv - mu_i_fv * i_fv

            # Recovered
            dr_uv_dt = gamma_i_uv * i_uv + \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv
            dr_pv_dt = gamma_i_pv * i_pv + - \
                self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_uv - self.epidemiological_data[
                           'percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv
            dr_fv_dt = gamma_i_fv * i_fv + \
                self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[
                           min(int(t), len(self.epidemiological_data) - 1)] * r_pv

            # Dead
            dd_uv_dt = mu_i_uv * i_uv
            dd_pv_dt = mu_i_pv * i_pv
            dd_fv_dt = mu_i_fv * i_fv

            return ds_uv_dt, ds_pv_dt, ds_fv_dt, de_uv_dt, de_pv_dt, de_fv_dt, di_uv_dt, di_pv_dt, di_fv_dt, \
                dr_uv_dt, dr_pv_dt, dr_fv_dt, dd_uv_dt, dd_pv_dt, dd_fv_dt

    def ode_solver(self, y0, t, population, parameters, solver='odeint', method='RK45',
                   differential_equations_version=1):
        """This function solves the ordinary differential equations for the epidemiological model.

        :parameter y0: Vector of initial population dynamics
        :parameter t: Time span of simulation
        :parameter population: Total Population
        :parameter parameters: Parameter values
        :parameter solver: String - Name of the solver
        :parameter method: Integration method used by the solver.
        :parameter differential_equations_version: Integer representing the model/differential equations we want to use.

        :return x_odeint: model predictions from Scipy's odeint method
        :return x_solve_ivp.y.T: model predictions form Scipy's solve_ivp method"""

        if solver == 'odeint':
            x_odeint = odeint(self.differential_equations, y0, t,
                              args=(population, parameters, True, differential_equations_version))
            # print(self.counter)
            # self.counter += 1
            return x_odeint

        elif solver == 'solve_ivp':
            x_solve_ivp = solve_ivp(self.differential_equations, y0=y0, t_span=(min(t), max(t)), t_eval=t,
                                    method=method, args=(population, parameters, False, differential_equations_version))
            # print(self.counter)
            # self.counter += 1
            return x_solve_ivp.y.T

    def residual(self, parameters, t, data, solver='odeint', method='RK45', differential_equations_version=1):
        """This function computes the residuals between the model predictions and the actual data.

        :parameter parameters: Parameter values
        :parameter t: Time span of simulation
        :parameter data: Real-world data we want to fit our model to
        :parameter solver: String - Name of the solver
        :parameter method: Integration method used by the solver.
        :parameter differential_equations_version: Integer representing the model/differential equations we want to use.

        :returns: residuals"""

        model_predictions = self.ode_solver(self.y0, t, self.population, parameters, solver=solver, method=method,
                                            differential_equations_version=differential_equations_version)

        model_predictions = pd.DataFrame(model_predictions, columns=self.compartment_names)

        residual = (model_predictions.values - data).ravel()

        return residual

    def residual_odeint(self, parameters, t, data, method='RK45', differential_equations_version=1):
        residual_odeint = self.residual(parameters, t, data, solver='odeint', method=method,
                                        differential_equations_version=differential_equations_version)
        return residual_odeint

    def residual_solve_ivp(self, parameters, t, data, method='RK45', differential_equations_version=1):
        residual_solve_ivp = self.residual(parameters, t, data, solver='solve_ivp', method=method,
                                           differential_equations_version=differential_equations_version)
        return residual_solve_ivp

    def plot(self, model_predictions):
        """This method plots the model predictions vs the actual data.

        :parameter model_predictions: Array - Model predictions."""

        for i, compartment_name in enumerate(self.compartment_names):
            plt.figure(figsize=(16, 10))
            plt.plot(self.t, self.epidemiological_compartment_values[:, i], linewidth=3, label=compartment_name)
            plt.plot(self.t, model_predictions[:, i], '--', linewidth=3, c='red', label='Best Fit ODE')
            plt.xlabel('Days', fontsize=24)
            plt.ylabel('Population', fontsize=24)
            plt.title(f'{compartment_name} vs Best Fit ODE', fontsize=32)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.show()


number_of_epidemiological_models = 4
minimization_methods = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping', 'ampgo',
                        'nelder', 'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs', 'tnc', 'trust-ncg',
                        'trust-exact', 'trust-krylov', 'trust-constr', 'dogleg', 'slsqp', 'emcee', 'shgo',
                        'dual_annealing']
integration_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

compartments = ['Susceptible_UV', 'Susceptible_PV', 'Susceptible_FV', 'Exposed_UV', 'Exposed_PV', 'Exposed_FV',
                'Infected_UV', 'Infected_PV', 'Infected_FV', 'Hospitalized_UV', 'Hospitalized_PV',
                'Hospitalized_FV', 'Recovered_UV', 'Recovered_PV', 'Recovered_FV', 'Dead_UV', 'Dead_PV', 'Dead_FV']

parameter_computation = ParameterComputation(compartment_names=compartments, constrained_beta=False)

# SOLVER = odeint
for minimization_method in minimization_methods:
    print(f'\n\n\nMinimization Method: {minimization_method}\n')
    start = time()
    model_fit_odeint = minimize(parameter_computation.residual_odeint, parameter_computation.parameters,
                                args=(parameter_computation.t, parameter_computation.epidemiological_compartment_values,
                                      'RK45', 1), method=minimization_method)
    print('Runtime:', time() - start, 'seconds')
    model_pred = parameter_computation.epidemiological_compartment_values + \
        model_fit_odeint.residual.reshape(parameter_computation.epidemiological_compartment_values.shape)
    print('Model Fit:\n', report_fit(model_fit_odeint))
    parameter_computation.plot(model_pred)

# SOLVER = solve_ivp
for minimization_method in minimization_methods:
    for integration_method in integration_methods:
        print(f'\n\n\nMinimization Method: {minimization_method}\n Integration Method: {integration_method}\n')
        start = time()
        model_fit_solve_ivp = minimize(
            parameter_computation.residual_solve_ivp, parameter_computation.parameters,
            args=(parameter_computation.t, parameter_computation.epidemiological_compartment_values,
                  integration_method, 4), method=minimization_method)
        print('Runtime:', time() - start, 'seconds')
        model_pred = parameter_computation.epidemiological_compartment_values + \
            model_fit_solve_ivp.residual.reshape(parameter_computation.epidemiological_compartment_values.shape)
        print('Model Fit:\n', report_fit(model_fit_solve_ivp))
        parameter_computation.plot(model_pred)

parameter_computation = ParameterComputation(filepath='./epidemiological_model_data_proportional_split.csv',
                                             compartment_names=compartments, constrained_beta=False, data_split=30)

start = time()
model_fit_solve_ivp = minimize(
    parameter_computation.residual_solve_ivp, parameter_computation.parameters,
    args=(parameter_computation.t, parameter_computation.epidemiological_compartment_values,
          'RK45', 5), method='leastsq')
print('Runtime:', time() - start, 'seconds')
model_pred = parameter_computation.epidemiological_compartment_values + \
    model_fit_solve_ivp.residual.reshape(parameter_computation.epidemiological_compartment_values.shape)
print('Model Fit:\n', report_fit(model_fit_solve_ivp))
parameter_computation.plot(model_pred)
