from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, List
import pandas as pd

WEIGHT_SCALES = {
    "severity": .4,
    "maintenance_effort": .2,
    "technical_maturity": .2,
    "improvement_opportunity": .2
}

class SeverityWeight(Enum):
    """
    Severity
    Very High 0.05: Failure and very likely safety issue
    High 0.5: Failure and likely safety issue
    Average 1: Failure and potential safety issue
    Low 3: Failure and less likely safety issue
    Very Low 4: Failure but no safety concern at all
    """
    VERY_HIGH = 0.05
    HIGH = 0.5
    AVERAGE = 1
    LOW = 3
    VERY_LOW = 4

class MaintenanceEffortWeight(Enum):
    """
    Maintenance Effort (cost, time, spare parts availability, etc.)
    Very High 0.05: Very high cost/long repair time/difficult to get spare parts needed
    High 1: High cost/long repair time/difficult to get spare parts needed
    Average 2: Average cost/long repair time/difficult to get spare parts needed
    Low 3: Low cost/long repair time/difficult to get spare parts needed
    Very Low 4: Very low cost/long repair time/difficult to get spare parts needed
    """
    VERY_HIGH = 0.05
    HIGH = 1
    AVERAGE = 2
    LOW = 3
    VERY_LOW = 4

class TechnicalMaturityWeight(Enum):
    """
    Technical Maturity
    Very High 0.05: Same technology used before in same application
    High 1: Similar technology used before un same application
    Average 2: Quite different application with same technology
    Low 3: New technology used in industry but not by Rivian
    Very Low 4: Unique technology never used in industry
    """
    VERY_HIGH = 0.05
    HIGH = 1
    AVERAGE = 2
    LOW = 3
    VERY_LOW = 4

class ImprovementOpportunityWeight(Enum):
    """
    Improvement Opportunity
    Very High 0.05: Very expensive or almost impossible to improve
    High 1: Expensive to improve
    Average 2: Average
    Low 3: Less expensive to improve
    Very Low 4: Least expensive to improve
    """
    VERY_HIGH = 0.05
    HIGH = 1
    AVERAGE = 2
    LOW = 3
    VERY_LOW = 4



# Equivalent MRR Allocation
def equivalent_mrr_allocation(lambda_g: float, component_n: int) -> float:
    """
    Equivalent MRR Allocation Equation
    :param lambda_g: Failure Rate System Target
    :param component_n: Count of Components/Subsystems to Distribute the Failure Rate
    :return: lambda_i = lambda_g / component_n
    """
    if component_n <= 0:
        raise ValueError('Number of Components/Subsystems must be greater than 0')
    return lambda_g / component_n


@dataclass
class ReferenceAllocation:
    reference_mrr: float
    reference_count: int = 1
    reference_take_rate: float = 1
    reference_allocation_percentage: float = field(init=False, default=0)
    reference_allocated_mrr: float = field(init=False, default=0)

    def __post_init__(self):
        if self.reference_count <= 0:
            raise ValueError('Number of Components/Subsystems must be greater than 0')
        if self.reference_take_rate <= 0 or self.reference_take_rate > 1:
            raise ValueError('Take Rate must be a percentage between 0 and 1')

    def to_dataframe(self):
        df =  pd.DataFrame([self])
        df['reference_take_rate'] = (df['reference_take_rate']*100).map('{:.0f}%'.format)
        df.columns = [x.replace('reference_', '') for x in df.columns]
        df.columns = pd.MultiIndex.from_product([['Reference']] + [df.columns])
        return df

@dataclass
class ReferenceToNewComponent:
    """
    Reference to New Component

    """
    content: float = 1
    usage: float = 1
    stress: float = 1
    content_exponent: int = 1
    usage_exponent: int = 1
    stress_exponent: int = 2
    multiplier: float = field(init=False)
    modified_allocation_percentage: float = field(init=False, default=0)
    modified_allocated_mrr: float = field(init=False, default=0)

    def __post_init__(self):
        self.multiplier = self.content ** self.content_exponent * self.usage ** self.usage_exponent * self.stress ** self.stress_exponent

    def to_dataframe(self):
        df =  pd.DataFrame([self])
        df.columns = pd.MultiIndex.from_product([['Reference to New Component']] + [df.columns])
        return df

@dataclass
class Ranking:
    severity: SeverityWeight
    maintenance_effort: MaintenanceEffortWeight
    technical_maturity: TechnicalMaturityWeight
    improvement_opportunity: ImprovementOpportunityWeight
    weighted_score: float = field(init=False)

    def __post_init__(self):
        self.weighted_score = (self.severity.value * WEIGHT_SCALES['severity'] +
                               self.maintenance_effort.value * WEIGHT_SCALES['maintenance_effort'] +
                               self.technical_maturity.value * WEIGHT_SCALES['technical_maturity'] +
                               self.improvement_opportunity.value * WEIGHT_SCALES['improvement_opportunity'])

    def to_dataframe(self):
        df =  pd.DataFrame([self])
        df['severity'] = self.severity.name
        df['maintenance_effort'] = self.maintenance_effort.name
        df['technical_maturity'] = self.technical_maturity.name
        df['improvement_opportunity'] = self.improvement_opportunity.name
        df.columns = pd.MultiIndex.from_product([['Scoring and Scaling']] + [df.columns])
        return df

@dataclass
class MRRAllocation:
    vehicle_super_component_mrr: float
    vehicle_component_mrr: float
    super_component_mrr: float
    component_mrr: float

    def to_dataframe(self):
        df =  pd.DataFrame([self])
        print(df)
        df.columns = [('Vehicle Level', x.replace('vehicle_', '')) if 'vehicle_' in x else ('Component Level', x) for x in df.columns]
        return df

@dataclass
class ComponentAllocation:
    name: str
    take_rate: float = 1
    count: int = 1
    preset_mrr: Optional[float] = None
    reference_allocation: Optional[ReferenceAllocation]  = None
    reference_to_new_component: Optional[ReferenceToNewComponent] = None
    ranking: Optional[Ranking]  = None
    method: str = field(init=False)
    output_mrr: Optional[MRRAllocation] = field(init=False, default=None)

    def __post_init__(self):
        if self.count <= 0:
            raise ValueError('Number of Components/Subsystems must be greater than 0')
        if self.take_rate <= 0 or self.take_rate > 1:
            raise ValueError('Take Rate must be a percentage between 0 and 1')
        if self.preset_mrr is not None:
            self.method = 'Preset MRR'
        else:
            self.method = 'Allocated MRR'

    def to_dataframe(self):
        df =  pd.DataFrame([self])
        df['take_rate'] = (df['take_rate'] * 100).map('{:.0f}%'.format)
        df.drop(columns=['output_mrr','reference_allocation', 'reference_to_new_component', 'ranking'], inplace=True)
        if self.reference_allocation is not None:
            df = pd.concat([df, self.reference_allocation.to_dataframe()], axis=1)
        if self.reference_to_new_component is not None:
            df = pd.concat([df, self.reference_to_new_component.to_dataframe()], axis=1)
        if self.ranking is not None:
            df = pd.concat([df, self.ranking.to_dataframe()], axis=1)
        if self.output_mrr is not None:
            df = pd.concat([df, self.output_mrr.to_dataframe()], axis=1)
        return df

@dataclass
class SystemAllocation:
    system_mrr_goal: float
    components: List[ComponentAllocation]
    reference_mrr: Optional[float] = None
    preset_mrr: float = field(init=False)
    remaining_mrr: float = field(init=False)
    

    def __post_init__(self):
        if self.system_mrr_goal <= 0:
            raise ValueError('System MRR Goal must be greater than 0')
        # Allocate Preset MRR to Components
        self.preset_components = [c for c in self.components if c.preset_mrr is not None]
        for c in self.preset_components:
            c.output_mrr = MRRAllocation(
                vehicle_super_component_mrr=c.preset_mrr,
                vehicle_component_mrr=c.preset_mrr / c.count,
                super_component_mrr=c.preset_mrr / c.take_rate,
                component_mrr=c.preset_mrr / c.take_rate / c.count
            )

        self.preset_mrr = sum(c.preset_mrr for c in self.preset_components)
        if self.preset_mrr > self.system_mrr_goal:
            raise ValueError('Preset MRR is greater than System MRR Goal')

        # Recalculate Remaining MRR
        self.remaining_mrr = self.system_mrr_goal - self.preset_mrr
        self.eval_components = [c for c in self.components if c.preset_mrr is None]

        # Equal MRR Allocation to Components without Reference
        self.no_reference_components = [c for c in self.eval_components if c.reference_allocation is None]
        count = sum(c.count for c in self.eval_components)
        equivalent_mrr_dist = equivalent_mrr_allocation(self.remaining_mrr, count)
        for c in self.no_reference_components:
            c.preset_mrr = equivalent_mrr_dist * c.count
            c.output_mrr = MRRAllocation(
                vehicle_super_component_mrr=c.preset_mrr,
                vehicle_component_mrr=c.preset_mrr/c.count,
                super_component_mrr=c.preset_mrr/c.take_rate,
                component_mrr=c.preset_mrr/ c.take_rate /c.count
            )

        # Recalculate Remaining MRR
        updated_preset_mrr = sum(c.preset_mrr for c in self.no_reference_components)
        self.remaining_mrr = self.remaining_mrr - updated_preset_mrr
        self.preset_mrr = self.preset_mrr + updated_preset_mrr

        # Initial MRR Allocation to Components with Reference
        self.reference_components = [c for c in self.eval_components if c.reference_allocation is not None]
        reference_mrr_sum = sum(c.reference_allocation.reference_mrr for c in self.reference_components)
        print(f'Reference MRR Sum: {reference_mrr_sum}')
        for c in self.reference_components:
            c.reference_allocation.reference_allocation_percentage = c.reference_allocation.reference_mrr / reference_mrr_sum
            c.reference_allocation.reference_allocated_mrr = c.reference_allocation.reference_allocation_percentage * self.remaining_mrr

        # Modified MRR Allocation to Components with Reference and Reference to New Component
        reference_to_new_sumprod = sum([c.reference_allocation.reference_allocation_percentage * c.reference_to_new_component.multiplier * (c.take_rate / c.reference_allocation.reference_take_rate) for c in self.reference_components])
        for c in self.reference_components:
            c.reference_to_new_component.modified_allocation_percentage = c.reference_allocation.reference_allocation_percentage * c.reference_to_new_component.multiplier * (c.take_rate / c.reference_allocation.reference_take_rate) / reference_to_new_sumprod
            c.reference_to_new_component.modified_allocated_mrr = c.reference_to_new_component.modified_allocation_percentage * self.remaining_mrr


        # Scoring and Scaling Adjustment
        weighted_score_sum_prod = sum([c.ranking.weighted_score * c.reference_to_new_component.modified_allocated_mrr for c in self.reference_components])
        scaling_factor = self.remaining_mrr / weighted_score_sum_prod
        for c in self.reference_components:
            c.output_mrr = MRRAllocation(
                vehicle_super_component_mrr= c.reference_to_new_component.modified_allocated_mrr * scaling_factor * c.ranking.weighted_score,
                vehicle_component_mrr= c.reference_to_new_component.modified_allocated_mrr * scaling_factor * c.ranking.weighted_score / c.count,
                super_component_mrr=c.reference_to_new_component.modified_allocated_mrr * scaling_factor * c.ranking.weighted_score / c.take_rate,
                component_mrr=c.reference_to_new_component.modified_allocated_mrr * scaling_factor * c.ranking.weighted_score / c.take_rate / c.count,
            )

    def create_full_datatable(self):
        output = [c.to_dataframe() for c in self.components]
        df = pd.concat(output, axis=0, ignore_index=True)
        df.columns = [('',x) if isinstance(x, str) else x for x in df.columns]
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        percentage_cols = [col for col in df.columns if 'percentage' in col[1]]
        for col in percentage_cols:
            df[col] = (df[col]*100).map('{:.2f}%'.format)
            df[col] = df[col].replace('nan%', None)
        rename_percentage_cols = {col[1]: col[1].replace('percentage', '%') for col in percentage_cols}

        df = df.rename(columns=rename_percentage_cols, level=1)
        df = df.rename(columns={col[1]: col[1].replace('_', ' ').title() for col in df.columns}, level=1)
        mrr_columns = [col[1] for col in df.columns if 'Mrr' in col[1]]
        df = df.rename(columns={col: col.replace('Mrr', 'MRR') for col in mrr_columns}, level=1)

        return df


    def create_mrr_datatable(self):
        df = self.create_full_datatable()
        df_components = df.filter(like='Name', axis=1)
        df_filtered = df.filter(like='Level', axis=1)
        df = pd.concat([df_components, df_filtered], axis=1)
        return df


if __name__ == '__main__':

    r1 = ReferenceAllocation(reference_mrr=50, reference_count=1, reference_take_rate=1)
    # print(r1.to_dataframe())

    c1 =  ComponentAllocation(name='Component 1', take_rate=1, count=1,
                                reference_allocation=ReferenceAllocation(reference_mrr=50, reference_count=1, reference_take_rate=1),
                                reference_to_new_component=ReferenceToNewComponent(),
                                ranking=Ranking(
                                    severity=SeverityWeight.VERY_HIGH,
                                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE))
    # print(c1.to_dataframe())

    s1 = SystemAllocation(
        system_mrr_goal=2000,
        components=[
            ComponentAllocation(
                name='Component 1',
                take_rate=1,
                count=1,
                reference_allocation=ReferenceAllocation(reference_mrr=50, reference_count=1, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.VERY_HIGH,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 2',
                take_rate=1,
                count=1,
                reference_allocation=ReferenceAllocation(reference_mrr=1000, reference_count=1, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 3',
                take_rate=1,
                count=1,
                reference_allocation=ReferenceAllocation(reference_mrr=50, reference_count=1, reference_take_rate=.5),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 4',
                take_rate=1,
                count=2,
                reference_allocation=ReferenceAllocation(reference_mrr=200, reference_count=2, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 5',
                take_rate=1,
                count=1,
                reference_allocation=ReferenceAllocation(reference_mrr=100, reference_count=1, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 6',
                take_rate=1,
                count=2,
                reference_allocation=ReferenceAllocation(reference_mrr=100, reference_count=2, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(
                name='Component 7',
                take_rate=1,
                count=1,
                reference_allocation=ReferenceAllocation(reference_mrr=30, reference_count=1, reference_take_rate=1),
                reference_to_new_component=ReferenceToNewComponent(),
                ranking=Ranking(
                    severity=SeverityWeight.AVERAGE,
                    maintenance_effort=MaintenanceEffortWeight.AVERAGE,
                    technical_maturity=TechnicalMaturityWeight.AVERAGE,
                    improvement_opportunity=ImprovementOpportunityWeight.AVERAGE
                )
            ),
            ComponentAllocation(name='Component 8', take_rate=1, count=1, preset_mrr=500),
            ComponentAllocation(name='Component 9', take_rate=1, count=1),
        ]
    )

    df = s1.create_full_datatable()
    # print(df)
    # df.to_csv('test.csv')
    # df.to_excel('test.xlsx')

    df = s1.create_mrr_datatable()
    print(df)
