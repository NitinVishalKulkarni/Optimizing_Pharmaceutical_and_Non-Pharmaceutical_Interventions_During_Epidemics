import sys

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)


class EpidemiologicalDataPreProcessing:
    """This class creates the epidemiological data to be used for computing the parameters of the epidemiological
     model."""

    def __init__(self, filepath='./New_York.csv', population=19_453_734):
        """This method loads the data for pre-processing.

        :parameter filepath: String - Filepath of the epidemic dataset.
        :parameter population: Integer - Population of the epidemic region."""

        self.epidemiological_data = pd.read_csv(filepath)
        self.epidemiological_data['date'] = pd.to_datetime(self.epidemiological_data['date'])
        self.epidemiological_data = self.epidemiological_data.iloc[79:474]
        self.epidemiological_data.reset_index(inplace=True)
        # print(self.epidemiological_data['date'])
        # print(len(self.epidemiological_data))

        # self.cases_deaths_by_vaccination = pd.read_csv('cases_deaths_by_vaccination.csv')
        # self.cases_deaths_by_vaccination['uv_to_fv_ratio'] = \
        #     (self.cases_deaths_by_vaccination['Unvaccinated with outcome'] /
        #      self.cases_deaths_by_vaccination['Vaccinated with outcome'])
        # self.cases_deaths_by_vaccination = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['Age group'] == 'all_ages_adj']
        # self.cases_deaths_by_vaccination = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['Vaccine product'] == 'all_types']

        # """Booster preprocessing"""
        # # Booster preprocessing:
        # self.cases_deaths_by_booster = pd.read_csv('cases_deaths_by_booster.csv')
        # self.cases_deaths_by_booster = \
        #     self.cases_deaths_by_booster.loc[self.cases_deaths_by_booster['age_group'] == 'all_ages']
        # self.cases_deaths_by_vaccination = \
        #     self.cases_deaths_by_booster.loc[self.cases_deaths_by_booster['vaccine_product'] == 'all_types']
        #
        # self.cases_by_booster = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['outcome'] == 'case']
        # self.deaths_by_booster = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['outcome'] == 'death']
        #
        # self.cases_by_booster.to_csv('cases_by_booster_.csv', index=False)
        # self.deaths_by_booster.to_csv('deaths_by_booster_.csv', index=False)

        # # Vaccination compartments.
        # self.epidemiological_data['unvaccinated_individuals'] = \
        #     population - self.epidemiological_data['people_vaccinated']
        # self.epidemiological_data['partially_vaccinated_individuals'] = \
        #     self.epidemiological_data['people_vaccinated'] - self.epidemiological_data['people_fully_vaccinated']
        # self.epidemiological_data['fully_vaccinated_individuals'] = \
        #     self.epidemiological_data['people_fully_vaccinated']
        # self.epidemiological_data['boosted_individuals'] = self.epidemiological_data['total_boosters']
        #
        # fv_to_uv = self.epidemiological_data['fully_vaccinated_individuals'] / \
        #            self.epidemiological_data['unvaccinated_individuals']
        # fv_to_pv = self.epidemiological_data['fully_vaccinated_individuals'] / \
        #     self.epidemiological_data['partially_vaccinated_individuals']
        # b_to_fv = self.epidemiological_data['boosted_individuals'] / \
        #     self.epidemiological_data['fully_vaccinated_individuals']
        # b_to_pv = self.epidemiological_data['boosted_individuals'] / \
        #           self.epidemiological_data['partially_vaccinated_individuals']
        # b_to_uv = self.epidemiological_data['boosted_individuals'] / \
        #           self.epidemiological_data['unvaccinated_individuals']
        # print(len(fv_to_pv))
        # self.epidemiological_data['fv_to_uv'] = fv_to_uv
        # self.epidemiological_data['fv_to_pv'] = fv_to_pv
        # self.epidemiological_data['b_to_fv'] = b_to_fv
        # self.epidemiological_data['b_to_pv'] = b_to_pv
        # self.epidemiological_data['b_to_uv'] = b_to_uv
        # test = self.epidemiological_data[['date', 'fv_to_uv', 'fv_to_pv', 'b_to_uv', 'b_to_pv', 'b_to_fv']]
        # test.to_csv('ratio.csv', index=False)

        # self.cases_by_vaccination = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['outcome'] == 'case']
        # self.deaths_by_vaccination = \
        #     self.cases_deaths_by_vaccination.loc[self.cases_deaths_by_vaccination['outcome'] == 'death']
        #
        # self.cases_by_vaccination['fv_to_pv_ratio'] = fv_to_pv
        # self.deaths_by_vaccination['fv_to_pv_ratio'] = fv_to_pv
        #
        # self.cases_by_vaccination.to_csv('cases_by_vaccination_.csv', index=False)
        # self.deaths_by_vaccination.to_csv('deaths_by_vaccination_.csv', index=False)
        # self.cases_deaths_by_vaccination['uv']

        # self.cases_deaths_by_vaccination
        # sys.exit()

        self.cases_by_vaccination = pd.read_csv('cases_by_vaccination_and_booster.csv').iloc[:395]
        self.deaths_by_vaccination = pd.read_csv('deaths_by_vaccination_and_booster.csv').iloc[:395]
        self.hospitalizations_by_vaccination = pd.read_csv('hospitalizations_by_vaccination_and_booster.csv').iloc[:395]
        print(len(self.cases_by_vaccination), len(self.deaths_by_vaccination), len(self.hospitalizations_by_vaccination))
        self.population = population

    def data_preprocessing(self):
        """This method pre-processes the data for the sub-compartments in the epidemiological model."""

        # Vaccination compartments.
        self.epidemiological_data['unvaccinated_individuals'] = \
            self.population - self.epidemiological_data['people_vaccinated']
        # self.epidemiological_data['partially_vaccinated_individuals'] = \
        #     self.epidemiological_data['people_vaccinated'] - self.epidemiological_data['people_fully_vaccinated']
        self.epidemiological_data['fully_vaccinated_individuals'] = \
            self.epidemiological_data['people_fully_vaccinated']
        self.epidemiological_data['boosted_individuals'] = self.epidemiological_data['total_boosters']

        # Computing the vaccination rates.
        self.epidemiological_data[['percentage_unvaccinated_to_fully_vaccinated',
                                  'percentage_fully_vaccinated_to_boosted']] = 0

        for i in range(1, len(self.epidemiological_data)):
            """Now we don't consider partially vaccinated individuals."""
            # self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated'].iloc[i] \
            #     = (self.epidemiological_data['unvaccinated_individuals'].iloc[i - 1]
            #        - self.epidemiological_data['unvaccinated_individuals'].iloc[i]) / \
            #     self.epidemiological_data['unvaccinated_individuals'].iloc[i - 1]

            self.epidemiological_data['percentage_unvaccinated_to_fully_vaccinated'].iloc[i] \
                = (self.epidemiological_data['unvaccinated_individuals'].iloc[i - 1]
                   - self.epidemiological_data['unvaccinated_individuals'].iloc[i]) / \
                  self.epidemiological_data['unvaccinated_individuals'].iloc[i - 1]

            # self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated'].iloc[i] \
            #     = ((self.epidemiological_data['fully_vaccinated_individuals'].iloc[i]
            #         - self.epidemiological_data['fully_vaccinated_individuals'].iloc[i - 1])
            #        + (self.epidemiological_data['total_boosters'].iloc[i]
            #           - self.epidemiological_data['total_boosters'].iloc[i - 1])) / \
            #     self.epidemiological_data['partially_vaccinated_individuals'].iloc[i - 1]

            self.epidemiological_data['percentage_fully_vaccinated_to_boosted'].iloc[i] \
                = (self.epidemiological_data['total_boosters'].iloc[i]
                   - self.epidemiological_data['total_boosters'].iloc[i - 1]) / \
                self.epidemiological_data['fully_vaccinated_individuals'].iloc[i - 1]

        # Exposed compartments.
        exposure_multiplier = 100 / 0.7  # We have a reference for this. (Cited > 700 times).
        self.epidemiological_data['Exposed'] = \
            (self.epidemiological_data['New Cases'] * exposure_multiplier).astype(int)
        # self.epidemiological_data['Exposed_UV'] = (
        #         self.epidemiological_data['New Cases'] * self.epidemiological_data['unvaccinated_individuals'] *
        #          / self.population).astype(int)
        # self.epidemiological_data['Exposed_PV'] = (
        #         self.epidemiological_data['New Cases'] * self.epidemiological_data['partially_vaccinated_individuals'] *
        #         exposure_multiplier / self.population).astype(int)
        # self.epidemiological_data['Exposed_FV'] = (
        #         self.epidemiological_data['New Cases'] * self.epidemiological_data['fully_vaccinated_individuals'] *
        #         exposure_multiplier / self.population).astype(int)
        # self.epidemiological_data['Exposed'] = self.epidemiological_data['Exposed_UV'] + self.epidemiological_data[
        #     'Exposed_PV'] + self.epidemiological_data['Exposed_FV']

        # Susceptible compartments.
        self.epidemiological_data['Susceptible'] = \
            self.population - self.epidemiological_data['Exposed'] - self.epidemiological_data['Active Cases'] - \
            self.epidemiological_data['Total Recovered'] - self.epidemiological_data['Total Deaths']

        # self.epidemiological_data['Susceptible_UV'] = (self.epidemiological_data['unvaccinated_individuals'] *
        #                                                self.epidemiological_data[
        #                                                    'Susceptible'] / self.population).astype(int)
        # self.epidemiological_data['Susceptible_PV'] = (self.epidemiological_data['partially_vaccinated_individuals'] *
        #                                                self.epidemiological_data[
        #                                                    'Susceptible'] / self.population).astype(int)
        # self.epidemiological_data['Susceptible_FV'] = (self.epidemiological_data['fully_vaccinated_individuals'] *
        #                                                self.epidemiological_data[
        #                                                    'Susceptible'] / self.population).astype(int)

        # Infected compartments.
        # government_data_skew = ((self.epidemiological_data['Active Cases'] -
        #                         self.epidemiological_data['inpatient_beds_used_covid']) * 0.615).astype(int)
        # proportional_vaccination_skew = \
        #     ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid']) *
        #      self.epidemiological_data['unvaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Infected_UV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                     government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid'])
             * (self.cases_by_vaccination['uv_mul'])).astype(int)
        self.epidemiological_data['Infected_UV'] = cdc_skew

        # government_data_skew = ((self.epidemiological_data['Active Cases'] -
        #                          self.epidemiological_data['inpatient_beds_used_covid']) * 0.039).astype(int)
        # proportional_vaccination_skew = \
        #     ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid']) *
        #      self.epidemiological_data['partially_vaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Infected_PV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                     government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid'])
             * (self.cases_by_vaccination['fv_mul'])).astype(int)
        self.epidemiological_data['Infected_FV'] = cdc_skew

        # government_data_skew = ((self.epidemiological_data['Active Cases'] -
        #                          self.epidemiological_data['inpatient_beds_used_covid']) * 0.346).astype(int)
        # proportional_vaccination_skew = \
        #     ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid']) *
        #      self.epidemiological_data['fully_vaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Infected_FV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                     government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            ((self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid'])
             * (self.cases_by_vaccination['b_mul'])).astype(int)
        self.epidemiological_data['Infected_BV'] = cdc_skew

        # Hospitalized compartments.
        # government_data_skew = (self.epidemiological_data['inpatient_beds_used_covid'] * 0.715).astype(int)
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['inpatient_beds_used_covid'] *
        #      self.epidemiological_data['unvaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Hospitalized_UV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            (self.epidemiological_data['inpatient_beds_used_covid']
             * self.hospitalizations_by_vaccination['uv_mul']).astype(int)
        self.epidemiological_data['Hospitalized_UV'] = cdc_skew

        # government_data_skew = (self.epidemiological_data['inpatient_beds_used_covid'] * 0.05).astype(int)
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['inpatient_beds_used_covid'] *
        #      self.epidemiological_data['partially_vaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Hospitalized_PV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            (self.epidemiological_data['inpatient_beds_used_covid']
             * self.hospitalizations_by_vaccination['fv_mul']).astype(int)
        self.epidemiological_data['Hospitalized_FV'] = cdc_skew

        # government_data_skew = (self.epidemiological_data['inpatient_beds_used_covid'] * 0.235).astype(int)
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['inpatient_beds_used_covid'] *
        #      self.epidemiological_data['fully_vaccinated_individuals'] / self.population).astype(int)
        # self.epidemiological_data['Hospitalized_FV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        cdc_skew = \
            (self.epidemiological_data['inpatient_beds_used_covid']
             * self.hospitalizations_by_vaccination['b_mul']).astype(int)
        self.epidemiological_data['Hospitalized_BV'] = cdc_skew

        # Recovered compartments.
        initial_recovered_unvaccinated_skew = \
            (self.epidemiological_data['Total Recovered'].iloc[0]
             * self.cases_by_vaccination['uv_mul'].iloc[0]).astype(int)
        # cdc_skew = (self.epidemiological_data['New Recoveries'] * self.cases_by_vaccination['uv_mul']).astype(int)
        # cdc_skew[0] = 0
        # cdc_skew = cdc_skew.cumsum()
        # cdc_skew = cdc_skew + initial_recovered_unvaccinated_skew
        # self.epidemiological_data['Recovered_UV'] = cdc_skew

        initial_recovered_fully_vaccinated_skew = \
            (self.epidemiological_data['Total Recovered'].iloc[0]
             * self.cases_by_vaccination['fv_mul'].iloc[0]).astype(int)
        # cdc_skew = (self.epidemiological_data['New Recoveries'] * self.cases_by_vaccination['pv_mul']).astype(int)
        # cdc_skew[0] = 0
        # cdc_skew = cdc_skew.cumsum()
        # cdc_skew = cdc_skew + initial_recovered_partially_vaccinated_skew
        # self.epidemiological_data['Recovered_PV'] = cdc_skew

        initial_recovered_booster_vaccinated_skew = \
            (self.epidemiological_data['Total Recovered'].iloc[0]
             * self.cases_by_vaccination['b_mul'].iloc[0]).astype(int)
        # cdc_skew = (self.epidemiological_data['New Recoveries'] * self.cases_by_vaccination['fv_mul']).astype(int)
        # cdc_skew[0] = 0
        # cdc_skew = cdc_skew.cumsum()
        # cdc_skew = cdc_skew + initial_recovered_fully_vaccinated_skew
        # self.epidemiological_data['Recovered_FV'] = cdc_skew

        # uv_to_pv = self.epidemiological_data['percentage_unvaccinated_to_partially_vaccinated']
        # pv_to_fv = self.epidemiological_data['percentage_partially_vaccinated_to_fully_vaccinated']

        uv_to_fv = self.epidemiological_data['percentage_unvaccinated_to_fully_vaccinated']
        fv_to_bv = self.epidemiological_data['percentage_fully_vaccinated_to_boosted']

        self.epidemiological_data[['Recovered_UV', 'Recovered_FV', 'Recovered_BV']] = 0
        self.epidemiological_data['Recovered_UV'].iloc[0] = initial_recovered_unvaccinated_skew
        self.epidemiological_data['Recovered_FV'].iloc[0] = initial_recovered_fully_vaccinated_skew
        self.epidemiological_data['Recovered_BV'].iloc[0] = initial_recovered_booster_vaccinated_skew

        for i in range(1, len(self.epidemiological_data)):
            self.epidemiological_data['Recovered_UV'].iloc[i] = \
                (self.epidemiological_data['Recovered_UV'].iloc[i - 1]
                 + self.epidemiological_data['New Recoveries'].iloc[i] * self.cases_by_vaccination['uv_mul'].iloc[i]
                 - uv_to_fv[i] * self.epidemiological_data['Recovered_UV'].iloc[i - 1]).astype(int)
            self.epidemiological_data['Recovered_FV'].iloc[i] = \
                (self.epidemiological_data['Recovered_FV'].iloc[i - 1]
                 + self.epidemiological_data['New Recoveries'].iloc[i] * self.cases_by_vaccination['fv_mul'].iloc[i]
                 + uv_to_fv[i] * self.epidemiological_data['Recovered_UV'].iloc[i - 1]
                 - fv_to_bv[i] * self.epidemiological_data['Recovered_FV'].iloc[i - 1]).astype(int)
            self.epidemiological_data['Recovered_BV'].iloc[i] = \
                (self.epidemiological_data['Recovered_BV'].iloc[i - 1]
                 + self.epidemiological_data['New Recoveries'].iloc[i] * self.cases_by_vaccination['b_mul'].iloc[i]
                 + fv_to_bv[i] * self.epidemiological_data['Recovered_FV'].iloc[i - 1]).astype(int)

        # Deceased compartments.
        # boolean_unvaccinated = 0.698 <= self.epidemiological_data['unvaccinated_individuals'].iloc[0] / self.population
        # government_data_skew = (self.epidemiological_data['Total Deaths'] * 0.698).astype(int)
        # self.epidemiological_data['Deceased_GOV_UV'] = government_data_skew
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['Total Deaths'] *
        #      self.epidemiological_data['unvaccinated_individuals'] / self.population).astype(int)
        # if boolean_unvaccinated:
        #     self.epidemiological_data['Deceased_UV'] = government_data_skew
        # else:
        #     self.epidemiological_data['Deceased_UV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        initial_deceased_skew = \
            (self.epidemiological_data['Total Deaths'].iloc[0]
             * self.cases_by_vaccination['uv_mul'].iloc[0]).astype(int)
        cdc_skew = (self.epidemiological_data['New Deaths'] * self.cases_by_vaccination['uv_mul']).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data['Deceased_UV'] = cdc_skew

        # boolean_partially_vaccinated = 0.051 <= self.epidemiological_data['partially_vaccinated_individuals'].iloc[
        #     0] / self.population
        # government_data_skew = (self.epidemiological_data['Total Deaths'] * 0.051).astype(int)
        # self.epidemiological_data['Deceased_GOV_PV'] = government_data_skew
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['Total Deaths'] *
        #      self.epidemiological_data['partially_vaccinated_individuals'] / self.population).astype(int)
        # if boolean_partially_vaccinated:
        #     self.epidemiological_data['Deceased_PV'] = government_data_skew
        # else:
        #     self.epidemiological_data['Deceased_PV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        initial_deceased_skew = \
            (self.epidemiological_data['Total Deaths'].iloc[0]
             * self.cases_by_vaccination['fv_mul'].iloc[0]).astype(int)
        cdc_skew = (self.epidemiological_data['New Deaths'] * self.cases_by_vaccination['fv_mul']).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data['Deceased_FV'] = cdc_skew

        # boolean_fully_vaccinated = 0.251 <= self.epidemiological_data['fully_vaccinated_individuals'].iloc[
        #     0] / self.population
        # government_data_skew = (self.epidemiological_data['Total Deaths'] * 0.251).astype(int)
        # self.epidemiological_data['Deceased_GOV_FV'] = government_data_skew
        # proportional_vaccination_skew = \
        #     (self.epidemiological_data['Total Deaths'] *
        #      self.epidemiological_data['fully_vaccinated_individuals'] / self.population).astype(int)
        # if boolean_fully_vaccinated:
        #     self.epidemiological_data['Deceased_FV'] = government_data_skew
        # else:
        #     self.epidemiological_data['Deceased_FV'] = np.where(government_data_skew <= proportional_vaccination_skew,
        #                                                         government_data_skew, proportional_vaccination_skew)
        initial_deceased_skew = \
            (self.epidemiological_data['Total Deaths'].iloc[0]
             * self.cases_by_vaccination['b_mul'].iloc[0]).astype(int)
        cdc_skew = (self.epidemiological_data['New Deaths'] * self.cases_by_vaccination['b_mul']).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data['Deceased_BV'] = cdc_skew

        # Accounting for "missing individuals".
        # missing_individuals_infected = \
        #     (self.epidemiological_data['Active Cases'] - self.epidemiological_data['inpatient_beds_used_covid']) - \
        #     (self.epidemiological_data['Infected_UV'] + self.epidemiological_data['Infected_PV'] +
        #      self.epidemiological_data['Infected_FV'])
        #
        # missing_individuals_hospitalized = \
        #     self.epidemiological_data['inpatient_beds_used_covid'] - \
        #     (self.epidemiological_data['Hospitalized_UV'] + self.epidemiological_data['Hospitalized_PV'] +
        #      self.epidemiological_data['Hospitalized_FV'])
        #
        # missing_individuals_recovered = \
        #     self.epidemiological_data['Total Recovered'] - \
        #     (self.epidemiological_data['Recovered_UV'] + self.epidemiological_data['Recovered_PV'] +
        #      self.epidemiological_data['Recovered_FV'])

        # missing_individuals_deceased = \
        #     self.epidemiological_data['Total Deaths'] - \
        #     (self.epidemiological_data['Deceased_UV'] + self.epidemiological_data['Deceased_PV'] +
        #      self.epidemiological_data['Deceased_FV'])

        missing_individuals_unvaccinated = \
            self.epidemiological_data['unvaccinated_individuals'] - \
            (
             self.epidemiological_data['Infected_UV'] + self.epidemiological_data['Hospitalized_UV'] +
             self.epidemiological_data['Recovered_UV'] + self.epidemiological_data['Deceased_UV'])

        missing_individuals_fully_vaccinated = \
            self.epidemiological_data['fully_vaccinated_individuals'] - \
            (
             self.epidemiological_data['Infected_FV'] + self.epidemiological_data['Hospitalized_FV'] +
             self.epidemiological_data['Recovered_FV'] + self.epidemiological_data['Deceased_FV'])

        missing_individuals_booster_vaccinated = \
            self.epidemiological_data['boosted_individuals'] - \
            (
             self.epidemiological_data['Infected_BV'] + self.epidemiological_data['Hospitalized_BV'] +
             self.epidemiological_data['Recovered_BV'] + self.epidemiological_data['Deceased_BV'])

        total_missing_individuals_vaccination = \
            missing_individuals_unvaccinated + missing_individuals_fully_vaccinated + \
            missing_individuals_booster_vaccinated

        # self.epidemiological_data['sum'] = self.epidemiological_data['Susceptible'] + self.epidemiological_data['Exposed']
        # self.epidemiological_data['mi'] = total_missing_individuals_vaccination
        # self.epidemiological_data['bool'] = self.epidemiological_data['sum'] >= self.epidemiological_data['mi']
        # print(self.epidemiological_data[['mi', 'sum', 'bool', 'Susceptible', 'Exposed']])
        # sys.exit()

        # # Adjusting the Infected compartment.
        # self.epidemiological_data['Infected_UV'] = \
        #     self.epidemiological_data['Infected_UV'] + \
        #     (missing_individuals_infected * missing_individuals_unvaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Infected_PV'] = \
        #     self.epidemiological_data['Infected_PV'] + \
        #     (missing_individuals_infected * missing_individuals_partially_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Infected_FV'] = \
        #     self.epidemiological_data['Infected_FV'] + \
        #     (missing_individuals_infected * missing_individuals_fully_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Infected'] = self.epidemiological_data['Infected_UV'] + self.epidemiological_data[
            'Infected_FV'] + self.epidemiological_data['Infected_BV']

        # # Adjusting the Hospitalized compartment.
        # self.epidemiological_data['Hospitalized_UV'] = \
        #     self.epidemiological_data['Hospitalized_UV'] + \
        #     (missing_individuals_hospitalized * missing_individuals_unvaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Hospitalized_PV'] = \
        #     self.epidemiological_data['Hospitalized_PV'] + \
        #     (missing_individuals_hospitalized * missing_individuals_partially_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Hospitalized_FV'] = \
        #     self.epidemiological_data['Hospitalized_FV'] + \
        #     (missing_individuals_hospitalized * missing_individuals_fully_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        self.epidemiological_data['Hospitalized'] = \
            self.epidemiological_data['Hospitalized_UV'] + self.epidemiological_data['Hospitalized_FV'] + \
            self.epidemiological_data['Hospitalized_BV']
        #
        # # Adjusting the Recovered compartment.
        # self.epidemiological_data['Recovered_UV'] = \
        #     self.epidemiological_data['Recovered_UV'] + \
        #     (missing_individuals_recovered * missing_individuals_unvaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Recovered_PV'] = \
        #     self.epidemiological_data['Recovered_PV'] + \
        #     (missing_individuals_recovered * missing_individuals_partially_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Recovered_FV'] = \
        #     self.epidemiological_data['Recovered_FV'] + \
        #     (missing_individuals_recovered * missing_individuals_fully_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        self.epidemiological_data['Recovered'] = self.epidemiological_data['Recovered_UV'] + \
            self.epidemiological_data['Recovered_FV'] + self.epidemiological_data['Recovered_BV']

        # Adjusting the Deceased compartment.
        # self.epidemiological_data['Deceased_UV'] = \
        #     self.epidemiological_data['Deceased_UV'] + \
        #     (missing_individuals_deceased * missing_individuals_unvaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Deceased_PV'] = \
        #     self.epidemiological_data['Deceased_PV'] + \
        #     (missing_individuals_deceased * missing_individuals_partially_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)
        #
        # self.epidemiological_data['Deceased_FV'] = \
        #     self.epidemiological_data['Deceased_FV'] + \
        #     (missing_individuals_deceased * missing_individuals_fully_vaccinated /
        #      total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Deceased'] = \
            self.epidemiological_data['Deceased_UV'] + self.epidemiological_data['Deceased_FV'] + \
            self.epidemiological_data['Deceased_BV']

        # Adjusting Susceptible
        self.epidemiological_data['Susceptible_UV'] = \
            (self.epidemiological_data['Susceptible'] * missing_individuals_unvaccinated
             / total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Susceptible_FV'] = \
            (self.epidemiological_data['Susceptible'] * missing_individuals_fully_vaccinated
             / total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Susceptible_BV'] = \
            (self.epidemiological_data['Susceptible'] * missing_individuals_booster_vaccinated
             / total_missing_individuals_vaccination).astype(int)

        # Adjusting Exposed
        self.epidemiological_data['Exposed_UV'] = \
            (self.epidemiological_data['Exposed'] * missing_individuals_unvaccinated
             / total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Exposed_FV'] = \
            (self.epidemiological_data['Exposed'] * missing_individuals_fully_vaccinated
             / total_missing_individuals_vaccination).astype(int)

        self.epidemiological_data['Exposed_BV'] = \
            (self.epidemiological_data['Exposed'] * missing_individuals_booster_vaccinated
             / total_missing_individuals_vaccination).astype(int)

        # Computing the total by vaccination statues across the different compartments.
        self.epidemiological_data['unvaccinated_compartment_total'] = \
            self.epidemiological_data['Susceptible_UV'] + self.epidemiological_data['Exposed_UV'] + \
            self.epidemiological_data['Infected_UV'] + self.epidemiological_data['Hospitalized_UV'] + \
            self.epidemiological_data['Recovered_UV'] + self.epidemiological_data['Deceased_UV']

        self.epidemiological_data['fully_vaccinated_compartment_total'] = \
            self.epidemiological_data['Susceptible_FV'] + self.epidemiological_data['Exposed_FV'] + \
            self.epidemiological_data['Infected_FV'] + self.epidemiological_data['Hospitalized_FV'] + \
            self.epidemiological_data['Recovered_FV'] + self.epidemiological_data['Deceased_FV']

        self.epidemiological_data['booster_vaccinated_compartment_total'] = \
            self.epidemiological_data['Susceptible_BV'] + self.epidemiological_data['Exposed_BV'] + \
            self.epidemiological_data['Infected_BV'] + self.epidemiological_data['Hospitalized_BV'] + \
            self.epidemiological_data['Recovered_BV'] + self.epidemiological_data['Deceased_BV']

        self.epidemiological_data['Original Infected'] = self.epidemiological_data['Active Cases'] - \
            self.epidemiological_data['inpatient_beds_used_covid']

        # Saving the epidemiological model data.
        self.epidemiological_data.iloc[:395].to_csv(
            'epidemiological_model_data_new.csv', index=False,
            columns=['date', 'unvaccinated_individuals', 'fully_vaccinated_individuals',
                     'boosted_individuals', 'unvaccinated_compartment_total',
                     'fully_vaccinated_compartment_total', 'booster_vaccinated_compartment_total',
                     'percentage_unvaccinated_to_fully_vaccinated',
                     'percentage_fully_vaccinated_to_boosted', 'New Cases',
                     'Susceptible', 'Exposed', 'Infected', 'Hospitalized', 'Recovered', 'Deceased',
                     'Original Infected', 'inpatient_beds_used_covid', 'Total Recovered', 'Total Deaths',
                     'Susceptible_UV', 'Susceptible_FV', 'Susceptible_BV', 'Exposed_UV', 'Exposed_FV', 'Exposed_BV',
                     'Infected_UV', 'Infected_FV', 'Infected_BV', 'Hospitalized_UV', 'Hospitalized_FV',
                     'Hospitalized_BV', 'Recovered_UV', 'Recovered_FV', 'Recovered_BV', 'Deceased_UV', 'Deceased_FV',
                     'Deceased_BV'])


epidemiological_data_preprocessing = EpidemiologicalDataPreProcessing()
epidemiological_data_preprocessing.data_preprocessing()
