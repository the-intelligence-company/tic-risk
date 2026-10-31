//! the texts behind the vocabulary ("names carry meaning"): every record kind, field, source,
//! unit, path and query target in swedish and english, and the code tables of the authorities'
//! codes (SNI, municipalities, status, case, role, cause and category codes). a code with no
//! table gets its key humanised and is counted by the build.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::fields::Unit;

pub struct KindText {
    pub key: &'static str,
    pub sv: &'static str,
    pub en: &'static str,
}

pub struct FieldText {
    pub key: &'static str,
    pub sv: &'static str,
    pub en: &'static str,
    pub unit: Unit,
}

const fn k(key: &'static str, sv: &'static str, en: &'static str) -> KindText {
    KindText { key, sv, en }
}

const fn f(key: &'static str, sv: &'static str, en: &'static str, unit: Unit) -> FieldText {
    FieldText { key, sv, en, unit }
}

use Unit::{Count, Days, Index, Months, None as NoUnit, Percent, Ratio, Sek};

/// record kinds: the event kinds, the derived kinds, the relation kinds, the state and context
/// kinds, the query kinds
pub const KINDS: &[KindText] = &[
    // insolvency (the labels)
    k(
        "bankruptcy.initiated",
        "Konkurs inledd",
        "bankruptcy initiated",
    ),
    k(
        "bankruptcy.terminated",
        "Konkurs avslutad",
        "bankruptcy terminated",
    ),
    k(
        "bankruptcy.repealed",
        "Konkurs upphävd av rätten",
        "bankruptcy repealed by the court",
    ),
    k(
        "liquidation.decided",
        "Likvidation beslutad",
        "liquidation decided",
    ),
    k(
        "liquidation.ended",
        "Likvidation avslutad",
        "liquidation ended",
    ),
    k(
        "reconstruction.decided",
        "Företagsrekonstruktion inledd",
        "company reconstruction decided",
    ),
    k(
        "reconstruction.ended",
        "Företagsrekonstruktion upphörd",
        "company reconstruction ended",
    ),
    k(
        "status.struck_off",
        "Bolaget avfört ur registret",
        "company struck off the register",
    ),
    // annual reports
    k(
        "report.received",
        "Årsredovisning inkommen till Bolagsverket",
        "annual report received by the registry",
    ),
    k(
        "report.approved",
        "Årsredovisning godkänd av Bolagsverket",
        "annual report approved by the registry",
    ),
    k(
        "report.registered",
        "Årsredovisning registrerad",
        "annual report registered",
    ),
    k(
        "report.shortage",
        "Föreläggande om brist i årsredovisningen",
        "deficiency notice on the annual report",
    ),
    k(
        "report.case_terminated",
        "Årsredovisningsärendet avslutat utan registrering",
        "annual-report case closed without registration",
    ),
    k(
        "fee.registered",
        "Förseningsavgift för årsredovisning påförd",
        "late-filing fee imposed",
    ),
    k(
        "fee.voided",
        "Förseningsavgift undanröjd",
        "late-filing fee voided",
    ),
    k(
        "dividend.registered",
        "Beslutad vinstutdelning registrerad",
        "dividend decision registered",
    ),
    k(
        "report.finding",
        "Iakttagelse i årsredovisningen",
        "finding in the annual report",
    ),
    k(
        "report.figures",
        "Årsredovisningens siffror (resultat- och balansräkning, nyckeltal)",
        "annual report figures (income statement, balance sheet, key ratios)",
    ),
    // registry enforcement
    k(
        "liquidation_order.issued",
        "Likvidationsföreläggande från Bolagsverket",
        "liquidation order issued by the registry",
    ),
    k(
        "liquidation_order.recalled",
        "Likvidationsföreläggande återkallat",
        "liquidation order recalled",
    ),
    // registry cases
    k(
        "case.received",
        "Ärende inkommet till Bolagsverket",
        "registry case received",
    ),
    k(
        "case.terminated",
        "Ärende hos Bolagsverket avslutat utan registrering",
        "registry case closed without registration",
    ),
    // board
    k(
        "board.gap_started",
        "Brist i styrelsen eller företrädarna uppstod",
        "board or representative gap started",
    ),
    k(
        "board.gap_ended",
        "Brist i styrelsen eller företrädarna åtgärdad",
        "board or representative gap ended",
    ),
    k(
        "board.composition",
        "Styrelsens sammansättning ändrad",
        "board composition changed",
    ),
    // other registry events
    k(
        "name.changed",
        "Företagsnamn ändrat",
        "company name changed",
    ),
    k(
        "purpose.changed",
        "Verksamhetsbeskrivning ändrad",
        "business purpose changed",
    ),
    k(
        "articles.changed",
        "Bolagsordning ändrad",
        "articles of association changed",
    ),
    // enforcement authority
    k(
        "debt.registered",
        "Skuld hos Kronofogden registrerad",
        "debt registered with the enforcement authority",
    ),
    k(
        "role.started",
        "Funktionär tillträdde",
        "an officer took up a role (board, CEO, auditor, signatory)",
    ),
    k(
        "role.ended",
        "Funktionär avgick",
        "an officer left a role",
    ),
    k(
        "ownership.owner_added",
        "Verklig huvudman tillkom",
        "a beneficial owner was added",
    ),
    k(
        "ownership.owner_removed",
        "Verklig huvudman upphörde",
        "a beneficial owner was removed",
    ),
    k(
        "payment_order.changed",
        "Betalningsföreläggande ändrat",
        "payment order changed (contested, withdrawn, dismissed or to court)",
    ),
    k(
        "payment_order.removed",
        "Betalningsföreläggande avfört",
        "payment order removed",
    ),
    k(
        "debt.present",
        "Skuld hos Kronofogden, startdag okänd",
        "debt with the enforcement authority, start unknown",
    ),
    k(
        "debt.changed",
        "Skuld hos Kronofogden ändrad",
        "debt with the enforcement authority changed",
    ),
    k(
        "debt.cleared",
        "Skuld hos Kronofogden reglerad",
        "debt with the enforcement authority cleared",
    ),
    k(
        "enforcement.case_registered",
        "Mål registrerat hos Kronofogden",
        "enforcement case registered",
    ),
    k(
        "enforcement.case_under_enforcement",
        "Mål under verkställighet hos Kronofogden",
        "enforcement case under execution",
    ),
    k(
        "enforcement.case_closed",
        "Mål hos Kronofogden avslutat",
        "enforcement case closed",
    ),
    k(
        "enforcement.case_partly_withdrawn",
        "Mål hos Kronofogden delvis återkallat",
        "enforcement case partly withdrawn",
    ),
    k(
        "enforcement.case_cancelled",
        "Mål hos Kronofogden makulerat",
        "enforcement case cancelled",
    ),
    k(
        "enforcement.no_assets_report",
        "Utredningsrapport: utmätningsbara tillgångar saknas",
        "enforcement report: no attachable assets",
    ),
    k(
        "enforcement.decision_cancelled",
        "Kronofogdens beslut makulerat",
        "enforcement decision cancelled",
    ),
    k(
        "payment_order.applied",
        "Ansökan om betalningsföreläggande mot företaget",
        "payment order applied for against the company",
    ),
    k(
        "payment_order.verdict",
        "Utslag i mål om betalningsföreläggande",
        "payment order verdict",
    ),
    k(
        "payment_order.withdrawn",
        "Ansökan om betalningsföreläggande återkallad",
        "payment order application withdrawn",
    ),
    k(
        "payment_order.dismissed",
        "Ansökan om betalningsföreläggande avskriven",
        "payment order application dismissed",
    ),
    k(
        "payment_order.to_court",
        "Mål om betalningsföreläggande överlämnat till tingsrätt",
        "payment order case referred to the district court",
    ),
    // tax agency
    k(
        "payroll.declared",
        "Arbetsgivardeklaration lämnad",
        "employer payroll declaration filed",
    ),
    k(
        "payroll.corrected",
        "Arbetsgivardeklaration rättad",
        "employer payroll declaration corrected",
    ),
    k(
        "tax.late_fee_return",
        "Förseningsavgift för deklaration",
        "late fee for a tax return",
    ),
    k(
        "tax.late_fee_income",
        "Förseningsavgift för inkomstdeklaration",
        "late fee for the income tax return",
    ),
    k(
        "tax.surcharge",
        "Skattetillägg påfört",
        "tax surcharge imposed",
    ),
    k(
        "tax.deferral",
        "Tillfälligt anstånd med skatt",
        "temporary tax deferral",
    ),
    k(
        "tax.registration",
        "Skatteregistrering ändrad (F-skatt, moms, arbetsgivare)",
        "tax registration changed (F-tax, VAT, employer)",
    ),
    // statistics office
    k(
        "industry.code",
        "Branschkod (SNI) registrerad",
        "industry code (SNI) registered",
    ),
    // permits
    k(
        "permit.serving_granted",
        "Serveringstillstånd beviljat",
        "serving permit granted",
    ),
    k(
        "permit.serving_ended",
        "Serveringstillstånd upphört",
        "serving permit ended",
    ),
    // relations
    k("role.board_member", "Styrelseledamot", "board member"),
    k(
        "role.deputy_board_member",
        "Styrelsesuppleant",
        "deputy board member",
    ),
    k("role.proprietor", "Innehavare", "proprietor"),
    k("role.chair", "Styrelseordförande", "chair of the board"),
    k("role.ceo", "Verkställande direktör", "chief executive"),
    k(
        "role.deputy_ceo",
        "Vice verkställande direktör",
        "deputy chief executive",
    ),
    k("role.partner", "Bolagsman", "partner"),
    k(
        "role.limited_partner",
        "Kommanditdelägare",
        "limited partner",
    ),
    k("role.liquidator", "Likvidator", "liquidator"),
    k("role.auditor", "Revisor", "auditor"),
    k(
        "role.auditor_in_charge",
        "Huvudansvarig revisor",
        "auditor in charge",
    ),
    k("role.deputy_auditor", "Revisorssuppleant", "deputy auditor"),
    k(
        "role.external_signatory",
        "Extern firmatecknare",
        "external signatory",
    ),
    k("role.other", "Annan funktionär", "other officer"),
    k(
        "ownership.beneficial",
        "Verklig huvudman",
        "beneficial owner",
    ),
    k(
        "ownership.beneficial_via",
        "Verklig huvudman genom annat företag",
        "beneficial owner through an intermediary company",
    ),
    k(
        "group.parent",
        "Moderbolag i koncernen",
        "parent company in the group",
    ),
    k(
        "address.registered",
        "Registrerad adress",
        "registered address",
    ),
    // derived state and context (source tic)
    k(
        "state.profile",
        "Företagets profil vid tidpunkten",
        "the company's profile at the as-of time",
    ),
    k(
        "state.interval",
        "Pågående tillstånd vid tidpunkten",
        "an active state at the as-of time",
    ),
    k(
        "state.filing",
        "Senaste årsredovisningen och räkenskapsåret",
        "the latest filing and the fiscal year",
    ),
    k(
        "state.sector",
        "Jämförelse med branschnyckeltal",
        "comparison with industry key figures",
    ),
    k(
        "state.coverage",
        "Källans täckning vid tidpunkten",
        "source coverage at the as-of time",
    ),
    k(
        "context.economy",
        "Ekonomisk serie, senaste publicerade värde",
        "economy series, latest published value",
    ),
    k(
        "context.industry",
        "Branschens produktionsserie, senaste publicerade värde",
        "industry production series, latest published value",
    ),
    k(
        "query",
        "Frågan som ställs till modellen",
        "the question put to the model",
    ),
];

/// fields: attribute names as family.attr, the figure concepts, the state and context fields
pub const FIELDS: &[FieldText] = &[
    // insolvency
    f(
        "bankruptcy.surplus",
        "Konkursen avslutades med överskott",
        "the bankruptcy ended with a surplus",
        NoUnit,
    ),
    f(
        "bankruptcy.initiated",
        "Dagar sedan konkursen inleddes",
        "days since the bankruptcy was initiated",
        Days,
    ),
    f(
        "liquidation.decider",
        "Vem beslutade likvidationen",
        "who decided the liquidation",
        NoUnit,
    ),
    f(
        "liquidation.compulsory",
        "Tvångslikvidation",
        "compulsory liquidation",
        NoUnit,
    ),
    f(
        "liquidation.decided",
        "Dagar sedan likvidationen beslutades",
        "days since the liquidation was decided",
        Days,
    ),
    f(
        "reconstruction.decided",
        "Dagar sedan rekonstruktionen inleddes",
        "days since the reconstruction was decided",
        Days,
    ),
    f(
        "status.status_type",
        "Avförandets statuskod",
        "status code of the striking off",
        NoUnit,
    ),
    // annual reports
    f(
        "report.period_start",
        "Räkenskapsårets början",
        "fiscal period start",
        Days,
    ),
    f(
        "report.period_end",
        "Räkenskapsårets slut",
        "fiscal period end",
        Days,
    ),
    f("report.interim", "Delårsrapport", "interim report", NoUnit),
    f(
        "report.consolidated",
        "Koncernredovisning",
        "consolidated accounts",
        NoUnit,
    ),
    f(
        "report.ruleset",
        "Redovisningsregelverk",
        "accounting framework",
        NoUnit,
    ),
    f(
        "report.arrival",
        "Dagar sedan handlingen kom in",
        "days since the filing arrived",
        Days,
    ),
    f(
        "report.code",
        "Bristkod i föreläggandet",
        "deficiency code of the notice",
        NoUnit,
    ),
    f(
        "report.respond_by",
        "Svarsfrist",
        "deadline to respond",
        Days,
    ),
    f(
        "report.reason",
        "Orsak till att ärendet avslutades",
        "reason the case was closed",
        NoUnit,
    ),
    f(
        "report.months",
        "Räkenskapsårets längd",
        "length of the fiscal period",
        Months,
    ),
    f(
        "report.source",
        "Siffrornas källa",
        "source of the figures",
        NoUnit,
    ),
    f(
        "report.precision",
        "Siffrornas noggrannhet",
        "precision of the figures",
        NoUnit,
    ),
    f(
        "report.remark_code",
        "Revisorns anmärkningskod",
        "auditor's remark code",
        NoUnit,
    ),
    f(
        "report.auditor_remark",
        "Revisorn har anmärkning",
        "auditor's report carries a remark",
        NoUnit,
    ),
    f(
        "report.finding",
        "Iakttagelsens art",
        "kind of finding",
        NoUnit,
    ),
    f(
        "report.score",
        "Iakttagelsens vikt",
        "weight of the finding",
        Count,
    ),
    f(
        "report.note_amount",
        "Belopp i iakttagelsen",
        "amount in the finding",
        Sek,
    ),
    f(
        "report.note_date",
        "Datum i iakttagelsen",
        "date in the finding",
        Days,
    ),
    f(
        "fee.number",
        "Förseningsavgiftens ordningsnummer (första, andra, tredje)",
        "late-fee number (first, second, third)",
        Count,
    ),
    f(
        "fee.period_end",
        "Räkenskapsår som avgiften gäller",
        "fiscal year the fee applies to",
        Days,
    ),
    f(
        "fee.legally_binding",
        "Dag då avgiften vinner laga kraft",
        "day the fee becomes legally binding",
        Days,
    ),
    f(
        "fee.cause",
        "Orsak till att avgiften undanröjdes",
        "cause of the fee being voided",
        NoUnit,
    ),
    f("dividend.type", "Utdelningens typ", "dividend type", NoUnit),
    f(
        "dividend.amount",
        "Utdelningsbelopp",
        "dividend amount",
        Sek,
    ),
    f(
        "dividend.currency",
        "Utdelningens valuta",
        "dividend currency",
        NoUnit,
    ),
    f(
        "dividend.period_end",
        "Räkenskapsår som utdelningen gäller",
        "fiscal year the dividend applies to",
        Days,
    ),
    f(
        "liquidation_order.cause",
        "Grund för föreläggandet",
        "cause of the order",
        NoUnit,
    ),
    f(
        "liquidation_order.reply_by",
        "Svarsfrist",
        "deadline to reply",
        Days,
    ),
    f(
        "liquidation_order.liquidation_on",
        "Dag då likvidation annars beslutas",
        "day liquidation is otherwise decided",
        Days,
    ),
    f("case.code", "Ärendekod", "case code", NoUnit),
    f("case.outcome", "Ärendets utfall", "case outcome", NoUnit),
    f("board.gap", "Bristens art", "kind of gap", NoUnit),
    f(
        "board.started",
        "Dagar sedan bristen uppstod",
        "days since the gap started",
        Days,
    ),
    f(
        "board.members",
        "Antal styrelseledamöter",
        "number of board members",
        Count,
    ),
    f(
        "board.deputies",
        "Antal styrelsesuppleanter",
        "number of deputy board members",
        Count,
    ),
    f("name.naming_type", "Namnets typ", "name type", NoUnit),
    f(
        "purpose.length",
        "Verksamhetsbeskrivningens längd",
        "length of the purpose text",
        Count,
    ),
    // enforcement authority
    f(
        "debt.case_type",
        "Måltyp (allmänt eller enskilt mål)",
        "case type (public or private claims)",
        NoUnit,
    ),
    f("debt.debt_type", "Skuldens medelslag", "debt type", NoUnit),
    f(
        "debt.amount",
        "Skuldbelopp hos Kronofogden",
        "debt amount with the enforcement authority",
        Sek,
    ),
    f("debt.cases", "Antal mål", "number of cases", Count),
    f(
        "debt.prev_amount",
        "Föregående skuldbelopp",
        "previous debt amount",
        Sek,
    ),
    f(
        "debt.prev_cases",
        "Föregående antal mål",
        "previous number of cases",
        Count,
    ),
    f(
        "enforcement.debt_type",
        "Målets medelslag",
        "debt type of the case",
        NoUnit,
    ),
    f(
        "enforcement.case_type",
        "Målets art",
        "kind of enforcement case",
        NoUnit,
    ),
    f(
        "enforcement.cause",
        "Orsak till avslut",
        "cause of closing",
        NoUnit,
    ),
    f(
        "enforcement.original_debt",
        "Ursprunglig skuld i målet",
        "original debt in the case",
        Sek,
    ),
    f(
        "enforcement.recalled_amount",
        "Återkallat belopp",
        "recalled amount",
        Sek,
    ),
    f(
        "enforcement.reason",
        "Skäl för beslutet",
        "reason for the decision",
        NoUnit,
    ),
    f(
        "payment_order.notification",
        "Aviseringens typ",
        "notification type",
        NoUnit,
    ),
    f(
        "payment_order.amount",
        "Yrkat belopp",
        "amount claimed",
        Sek,
    ),
    f(
        "payment_order.applicant_is_person",
        "Sökanden är en privatperson",
        "the applicant is a private person",
        NoUnit,
    ),
    f(
        "payment_order.contested",
        "Ansökan bestridd",
        "application contested",
        NoUnit,
    ),
    f(
        "payment_order.removal_cause",
        "Orsak till borttag",
        "cause of removal",
        NoUnit,
    ),
    // tax agency
    f(
        "payroll.period",
        "Deklarerad period",
        "declared period",
        Months,
    ),
    f(
        "payroll.amount",
        "Deklarerad bruttolön",
        "declared gross payroll",
        Sek,
    ),
    f(
        "payroll.prev_amount",
        "Tidigare deklarerad bruttolön",
        "previously declared gross payroll",
        Sek,
    ),
    f(
        "tax.declaration_day",
        "Deklarationsdag",
        "declaration day",
        Days,
    ),
    f("tax.amount", "Avgiftens belopp", "amount of the fee", Sek),
    f(
        "tax.period",
        "Deklarationsperiod",
        "declaration period",
        NoUnit,
    ),
    f(
        "tax.period_months",
        "Månader sedan deklarationsperiodens år",
        "months since the declaration period's year",
        Months,
    ),
    f("tax.kind", "Skatteslag", "kind of tax", NoUnit),
    f(
        "tax.period_from",
        "Periodens början",
        "period start",
        Months,
    ),
    f("tax.period_to", "Periodens slut", "period end", Months),
    f("tax.month", "Anståndsmånad", "deferral month", Months),
    f(
        "tax.outstanding",
        "Utestående anstånd",
        "outstanding deferral",
        Sek,
    ),
    f("tax.granted", "Beviljat anstånd", "deferral granted", Sek),
    f(
        "tax.withdrawn",
        "Återkallat anstånd",
        "deferral withdrawn",
        Sek,
    ),
    f("tax.ceased", "Upphört anstånd", "deferral ceased", Sek),
    f("tax.fskatt", "F-skattestatus", "F-tax status", NoUnit),
    f(
        "tax.fskatt_ended",
        "F-skattens slutdag",
        "end day of the F-tax approval",
        Days,
    ),
    f(
        "tax.reason",
        "Grund för återkallelse av F-skatt",
        "cause of the F-tax recall",
        NoUnit,
    ),
    f(
        "tax.vat",
        "Registrerad för moms",
        "registered for VAT",
        NoUnit,
    ),
    f(
        "tax.employer",
        "Registrerad som arbetsgivare",
        "registered as an employer",
        NoUnit,
    ),
    // statistics office
    f(
        "industry.sni",
        "Branschkod SNI 2025",
        "industry code SNI 2025",
        NoUnit,
    ),
    f(
        "industry.rank",
        "Branschkodens rang",
        "rank of the industry code",
        Count,
    ),
    f(
        "industry.prob",
        "Sannolikhet för den härledda branschkoden",
        "probability of the inferred industry code",
        Ratio,
    ),
    f(
        "industry.prob_division",
        "Sannolikhet för den härledda huvudgruppen",
        "probability of the inferred division",
        Ratio,
    ),
    f(
        "industry.abstain",
        "Branschtexten ger ingen bransch",
        "the purpose text gives no sector",
        NoUnit,
    ),
    // permits
    f(
        "permit.category",
        "Tillståndets kategori",
        "permit category",
        NoUnit,
    ),
    f(
        "permit.municipality",
        "Tillståndets kommun",
        "municipality of the permit",
        NoUnit,
    ),
    f(
        "permit.granted",
        "Dagar sedan tillståndet beviljades",
        "days since the permit was granted",
        Days,
    ),
    // relations
    f("role.code", "Funktionskod", "role code", NoUnit),
    f(
        "role.auditor_type",
        "Revisorns behörighet",
        "auditor qualification",
        NoUnit,
    ),
    f(
        "ownership.extent",
        "Ägandets omfattning",
        "extent of ownership",
        NoUnit,
    ),
    f(
        "ownership.govern",
        "Kontrollens art",
        "nature of control",
        NoUnit,
    ),
    f("group.share", "Ägarandel", "ownership share", Percent),
    f("address.type", "Adresstyp", "address type", NoUnit),
    f(
        "address.municipality",
        "Adressens kommun",
        "municipality of the address",
        NoUnit,
    ),
    // figures (the canonical set)
    f("figure.net_sales", "Nettoomsättning", "net sales", Sek),
    f(
        "figure.other_operating_income",
        "Övriga rörelseintäkter",
        "other operating income",
        Sek,
    ),
    f(
        "figure.operating_income",
        "Summa rörelseintäkter",
        "total operating income",
        Sek,
    ),
    f(
        "figure.raw_materials",
        "Råvaror och förnödenheter",
        "raw materials and consumables",
        Sek,
    ),
    f(
        "figure.goods_for_resale",
        "Handelsvaror",
        "goods for resale",
        Sek,
    ),
    f(
        "figure.other_external_costs",
        "Övriga externa kostnader",
        "other external costs",
        Sek,
    ),
    f(
        "figure.personnel_costs",
        "Personalkostnader",
        "personnel costs",
        Sek,
    ),
    f(
        "figure.depreciation",
        "Avskrivningar",
        "depreciation and amortisation",
        Sek,
    ),
    f(
        "figure.operating_expenses",
        "Summa rörelsekostnader",
        "total operating expenses",
        Sek,
    ),
    f(
        "figure.operating_result",
        "Rörelseresultat",
        "operating result",
        Sek,
    ),
    f(
        "figure.financial_income",
        "Finansiella intäkter",
        "financial income",
        Sek,
    ),
    f(
        "figure.financial_expenses",
        "Finansiella kostnader",
        "financial expenses",
        Sek,
    ),
    f(
        "figure.financial_items",
        "Finansnetto",
        "net financial items",
        Sek,
    ),
    f(
        "figure.result_after_financial",
        "Resultat efter finansiella poster",
        "result after financial items",
        Sek,
    ),
    f(
        "figure.appropriations",
        "Bokslutsdispositioner",
        "appropriations",
        Sek,
    ),
    f(
        "figure.result_before_tax",
        "Resultat före skatt",
        "result before tax",
        Sek,
    ),
    f(
        "figure.tax",
        "Skatt på årets resultat",
        "tax on the result",
        Sek,
    ),
    f("figure.net_result", "Årets resultat", "net result", Sek),
    f(
        "figure.intangible_assets",
        "Immateriella anläggningstillgångar",
        "intangible assets",
        Sek,
    ),
    f(
        "figure.tangible_assets",
        "Materiella anläggningstillgångar",
        "tangible assets",
        Sek,
    ),
    f(
        "figure.financial_assets",
        "Finansiella anläggningstillgångar",
        "financial assets",
        Sek,
    ),
    f(
        "figure.fixed_assets",
        "Summa anläggningstillgångar",
        "total fixed assets",
        Sek,
    ),
    f("figure.inventories", "Varulager", "inventories", Sek),
    f(
        "figure.accounts_receivable",
        "Kundfordringar",
        "accounts receivable",
        Sek,
    ),
    f(
        "figure.current_receivables",
        "Summa kortfristiga fordringar",
        "total current receivables",
        Sek,
    ),
    f(
        "figure.current_investments",
        "Kortfristiga placeringar",
        "current investments",
        Sek,
    ),
    f("figure.cash", "Kassa och bank", "cash and bank", Sek),
    f(
        "figure.current_assets",
        "Summa omsättningstillgångar",
        "total current assets",
        Sek,
    ),
    f(
        "figure.total_assets",
        "Summa tillgångar",
        "total assets",
        Sek,
    ),
    f("figure.share_capital", "Aktiekapital", "share capital", Sek),
    f(
        "figure.unrestricted_equity",
        "Fritt eget kapital",
        "unrestricted equity",
        Sek,
    ),
    f("figure.equity", "Summa eget kapital", "total equity", Sek),
    f(
        "figure.untaxed_reserves",
        "Obeskattade reserver",
        "untaxed reserves",
        Sek,
    ),
    f(
        "figure.long_term_bank_debt",
        "Långfristiga skulder till kreditinstitut",
        "long-term bank debt",
        Sek,
    ),
    f(
        "figure.long_term_liabilities",
        "Summa långfristiga skulder",
        "total long-term liabilities",
        Sek,
    ),
    f(
        "figure.short_term_bank_debt",
        "Kortfristiga skulder till kreditinstitut",
        "short-term bank debt",
        Sek,
    ),
    f(
        "figure.accounts_payable",
        "Leverantörsskulder",
        "accounts payable",
        Sek,
    ),
    f(
        "figure.tax_liabilities",
        "Skatteskulder",
        "tax liabilities",
        Sek,
    ),
    f(
        "figure.short_term_liabilities",
        "Summa kortfristiga skulder",
        "total short-term liabilities",
        Sek,
    ),
    f(
        "figure.total_equity_liabilities",
        "Summa eget kapital och skulder",
        "total equity and liabilities",
        Sek,
    ),
    f(
        "figure.employees",
        "Medelantal anställda",
        "average number of employees",
        Count,
    ),
    f(
        "figure.floating_charges",
        "Företagsinteckningar",
        "floating charges",
        Sek,
    ),
    f(
        "figure.property_mortgages",
        "Fastighetsinteckningar",
        "property mortgages",
        Sek,
    ),
    f(
        "figure.securities",
        "Ställda säkerheter",
        "pledged securities",
        Sek,
    ),
    f(
        "figure.contingent_liabilities",
        "Ansvarsförbindelser",
        "contingent liabilities",
        Sek,
    ),
    f(
        "figure.overdraft_granted",
        "Beviljad checkräkningskredit",
        "overdraft facility granted",
        Sek,
    ),
    f(
        "figure.dividend",
        "Föreslagen utdelning",
        "proposed dividend",
        Sek,
    ),
    f(
        "figure.operating_margin",
        "Rörelsemarginal",
        "operating margin",
        Ratio,
    ),
    f(
        "figure.gross_margin",
        "Bruttomarginal",
        "gross margin",
        Ratio,
    ),
    f("figure.net_margin", "Nettomarginal", "net margin", Ratio),
    f(
        "figure.roe",
        "Avkastning på eget kapital",
        "return on equity",
        Ratio,
    ),
    f(
        "figure.roa",
        "Avkastning på totalt kapital",
        "return on assets",
        Ratio,
    ),
    f("figure.equity_ratio", "Soliditet", "equity ratio", Ratio),
    f(
        "figure.debt_equity",
        "Skuldsättningsgrad",
        "debt-to-equity ratio",
        Ratio,
    ),
    f(
        "figure.quick_ratio",
        "Kassalikviditet",
        "quick ratio",
        Ratio,
    ),
    f(
        "figure.net_sales_change",
        "Nettoomsättningens förändring mot föregående år",
        "change in net sales against the previous year",
        Ratio,
    ),
    f(
        "figure.operating_result_change",
        "Rörelseresultatets förändring mot föregående år",
        "change in operating result against the previous year",
        Ratio,
    ),
    // state: profile
    f("profile.legal_form", "Företagsform", "legal form", NoUnit),
    f(
        "profile.age_months",
        "Företagets ålder",
        "company age",
        Months,
    ),
    f(
        "profile.sni",
        "Branschkod SNI 2025",
        "industry code SNI 2025",
        NoUnit,
    ),
    f(
        "profile.section",
        "Branschavdelning",
        "industry section",
        NoUnit,
    ),
    f(
        "profile.size_band",
        "Storleksklass efter antal anställda",
        "size band by employees",
        NoUnit,
    ),
    f(
        "profile.region",
        "Säteskommun",
        "municipality of the seat",
        NoUnit,
    ),
    f(
        "profile.events_known",
        "Antal kända händelser",
        "number of known events",
        Count,
    ),
    f(
        "profile.relations_known",
        "Antal kända relationer",
        "number of known relations",
        Count,
    ),
    f(
        "profile.address_cohabitants",
        "Antal företag på samma adress",
        "companies at the same address",
        Count,
    ),
    // state: intervals
    f("interval.name", "Tillståndets art", "kind of state", NoUnit),
    f(
        "interval.days",
        "Dagar i tillståndet",
        "days in the state",
        Days,
    ),
    f(
        "interval.amount",
        "Belopp i tillståndet",
        "amount in the state",
        Sek,
    ),
    f(
        "interval.cases",
        "Antal mål i tillståndet",
        "cases in the state",
        Count,
    ),
    f(
        "interval.code",
        "Tillståndets kod",
        "code of the state",
        NoUnit,
    ),
    // state: filing
    f(
        "filing.age_months",
        "Senaste årsredovisningens ålder",
        "age of the latest filing",
        Months,
    ),
    f(
        "filing.fye_month",
        "Räkenskapsårets slutmånad",
        "fiscal year-end month",
        NoUnit,
    ),
    f(
        "filing.months_to_deadline",
        "Månader till nästa inlämningsfrist",
        "months to the next filing deadline",
        Months,
    ),
    f(
        "filing.reports_known",
        "Antal kända årsredovisningar",
        "filings known",
        Count,
    ),
    f(
        "filing.loss_streak",
        "Förlustår i följd",
        "consecutive loss years",
        Count,
    ),
    f(
        "filing.equity_ratio_adj",
        "Justerad soliditet (79,4 % av obeskattade reserver)",
        "adjusted equity ratio (79.4 % of untaxed reserves)",
        Ratio,
    ),
    f(
        "filing.cash_ratio",
        "Kassa i förhållande till tillgångar",
        "cash to total assets",
        Ratio,
    ),
    f(
        "filing.short_debt_ratio",
        "Kortfristiga skulder i förhållande till tillgångar",
        "short-term liabilities to total assets",
        Ratio,
    ),
    f(
        "filing.bank_debt_ratio",
        "Bankskulder i förhållande till tillgångar",
        "bank debt to total assets",
        Ratio,
    ),
    f(
        "filing.interest_coverage",
        "Räntetäckningsgrad",
        "interest coverage",
        Ratio,
    ),
    f(
        "filing.loss",
        "Förlust senaste året",
        "loss in the latest year",
        NoUnit,
    ),
    f(
        "filing.negative_equity",
        "Negativt eget kapital",
        "negative equity",
        NoUnit,
    ),
    f(
        "filing.equity_below_half_capital",
        "Eget kapital under halva aktiekapitalet",
        "equity below half the share capital",
        NoUnit,
    ),
    f(
        "filing.overdue",
        "Årsredovisning saknas för ett förfallet räkenskapsår",
        "annual report overdue",
        NoUnit,
    ),
    f(
        "filing.overdue_months",
        "Månader sedan inlämningsfristen gick ut",
        "months since the filing deadline passed",
        Months,
    ),
    // state: sector
    f(
        "sector.figure",
        "Branschnyckeltal",
        "industry key figure",
        NoUnit,
    ),
    f(
        "sector.q1",
        "Branschens första kvartil",
        "industry first quartile",
        NoUnit,
    ),
    f(
        "sector.median",
        "Branschens median",
        "industry median",
        NoUnit,
    ),
    f(
        "sector.q3",
        "Branschens tredje kvartil",
        "industry third quartile",
        NoUnit,
    ),
    f(
        "sector.own",
        "Företagets eget värde",
        "the company's own value",
        NoUnit,
    ),
    f(
        "sector.diff",
        "Avvikelse från branschmedianen",
        "difference to the industry median",
        NoUnit,
    ),
    f(
        "sector.position",
        "Läge mot branschens kvartiler",
        "position against the industry quartiles",
        NoUnit,
    ),
    f(
        "sector.band",
        "Storleksklass i statistiken",
        "size band in the statistics",
        NoUnit,
    ),
    // state: coverage
    f("coverage.source", "Källa", "source", NoUnit),
    f(
        "coverage.state",
        "Källans tillstånd",
        "source state",
        NoUnit,
    ),
    f(
        "coverage.days_live",
        "Dagar sedan källan började",
        "days since the source went live",
        Days,
    ),
    // state: calendar
    f("calendar.month", "Kalendermånad", "calendar month", NoUnit),
    f(
        "calendar.days_to_deadline",
        "Dagar till nästa inlämningsfrist",
        "days to the next filing deadline",
        Days,
    ),
    f(
        "calendar.days_since_fye",
        "Dagar sedan senaste räkenskapsårsslut",
        "days since the latest fiscal year-end",
        Days,
    ),
    // context
    f(
        "economy.change_12m",
        "Förändring på tolv månader",
        "twelve-month change",
        Index,
    ),
    f(
        "economy.period",
        "Seriens senaste period",
        "latest period of the series",
        Months,
    ),
    // query
    f("query.target", "Frågans mål", "question target", NoUnit),
    f(
        "query.horizon_months",
        "Frågans horisont",
        "question horizon",
        Months,
    ),
    f(
        "query.as_of_month",
        "Frågans kalendermånad",
        "calendar month of the question",
        NoUnit,
    ),
];

/// sources; the record's `source` byte is the position here
pub const SOURCES: &[(&str, &str, &str)] = &[
    (
        "bolagsverket",
        "Bolagsverket",
        "the Swedish Companies Registration Office",
    ),
    (
        "kronofogden",
        "Kronofogden",
        "the Swedish Enforcement Authority",
    ),
    ("skatteverket", "Skatteverket", "the Swedish Tax Agency"),
    ("scb", "Statistiska centralbyrån", "Statistics Sweden"),
    ("court", "Domstol", "court"),
    (
        "folkhalsomyndigheten",
        "Folkhälsomyndigheten",
        "the Public Health Agency of Sweden",
    ),
    ("riksbank", "Riksbanken", "the Swedish central bank"),
    ("tic", "Härlett av TIC", "derived by TIC"),
];

pub fn source_index(code: &str) -> u8 {
    SOURCES
        .iter()
        .position(|(k, _, _)| *k == code)
        .and_then(|i| u8::try_from(i).ok())
        .unwrap_or(0)
}

pub const UNITS: &[(&str, &str, &str)] = &[
    ("none", "utan enhet", "no unit"),
    ("sek", "kronor", "Swedish kronor"),
    ("ratio", "andel", "ratio"),
    ("percent", "procent", "percent"),
    ("count", "antal", "count"),
    ("days", "dagar", "days"),
    ("months", "månader", "months"),
    ("year", "år", "year"),
    ("index", "indexvärde", "index value"),
];

pub const PATHS: &[(&str, &str, &str)] = &[
    ("focal", "företaget självt", "the company itself"),
    ("parent", "moderbolaget", "the parent company"),
    (
        "grandparent",
        "moderbolagets moderbolag",
        "the parent's parent",
    ),
    ("child", "dotterbolag", "a subsidiary"),
    ("sibling", "systerbolag", "a sister company"),
    (
        "owner",
        "ägarbolag (verklig huvudman genom bolag)",
        "an owner company (beneficial ownership through a company)",
    ),
    (
        "owned",
        "bolag som ägs genom företaget",
        "a company owned through the company",
    ),
    (
        "cohabitant",
        "företag på samma adress",
        "a company at the same address",
    ),
    ("auditor", "revisionsbyrån", "the audit firm"),
    (
        "via_officer",
        "annat företag med samma företrädare",
        "another company of the same officer",
    ),
];

pub const QUERIES: &[(&str, &str, &str)] = &[
    (
        "bankruptcy",
        "konkurs inom horisonten",
        "bankruptcy within the horizon",
    ),
    (
        "compulsory_liquidation",
        "tvångslikvidation inom horisonten",
        "compulsory liquidation within the horizon",
    ),
    (
        "reconstruction",
        "företagsrekonstruktion inom horisonten",
        "company reconstruction within the horizon",
    ),
    (
        "deregistration",
        "avregistrering inom horisonten",
        "deregistration within the horizon",
    ),
    (
        "any_default",
        "någon fallissemangshändelse inom horisonten",
        "any default-type event within the horizon",
    ),
    ("next_event", "nästa händelse", "the next event"),
    (
        "pretrain",
        "förträning utan fråga",
        "pretraining, no question",
    ),
    (
        "enforcement_debt",
        "skuld hos Kronofogden uppstår eller ökar inom horisonten",
        "enforcement debt arising or rising within the horizon",
    ),
    (
        "fskatt_revoked",
        "F-skattegodkännande återkallat inom horisonten",
        "F-tax approval revoked within the horizon",
    ),
    (
        "report_overdue",
        "förseningsavgift för årsredovisning inom horisonten",
        "late-filing fee for the annual report within the horizon",
    ),
];

/// the categorical values of the derived state records, interned at vocabulary creation so that
/// sampling never adds to the vocabulary; industry, municipality, debt type, gap kind and tax
/// registration cells reuse the event codes of the same values
pub const STATE_CODES: &[(&str, &[&str])] = &[
    ("profile.legal_form", &["AB", "HB", "KB", "EF"]),
    (
        "profile.section",
        &[
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
            "R", "S", "T", "U",
        ],
    ),
    (
        "profile.size_band",
        &[
            "unknown", "0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+",
        ],
    ),
    (
        "interval.name",
        &[
            "board_gap",
            "kfm_debt",
            "fskatt",
            "vat",
            "employer",
            "liquidation",
            "reconstruction",
            "liquidation_order",
            "tax_deferral",
            "serving_permit",
            "open_cases",
        ],
    ),
    (
        "sector.figure",
        &[
            "DebtEquityRatio",
            "EquityRatio",
            "NetMargin",
            "NetSalesPerEmployee",
            "OperatingMargin",
            "QuickRatio",
            "ReturnOnAssets",
            "ReturnOnEquity",
        ],
    ),
    ("sector.position", &["below_q1", "within", "above_q3"]),
    (
        "sector.band",
        &[
            "all", "0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+",
        ],
    ),
    ("coverage.state", &["live", "not_yet", "ended", "undated"]),
    (
        "calendar.month",
        &[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ],
    ),
    (
        "filing.fye_month",
        &[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ],
    ),
    (
        "query.as_of_month",
        &[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ],
    ),
    (
        "query.target",
        &[
            "bankruptcy",
            "compulsory_liquidation",
            "reconstruction",
            "deregistration",
            "any_default",
            "next_event",
            "pretrain",
            "enforcement_debt",
            "fskatt_revoked",
            "report_overdue",
        ],
    ),
];

#[derive(Deserialize)]
struct CodeRow {
    code: String,
    text: String,
    #[serde(default)]
    text_en: Option<String>,
    #[serde(default)]
    status_type: Option<String>,
}

struct Table {
    by_code: HashMap<String, (String, Option<String>)>,
}

fn table(json: &'static str) -> Table {
    let rows: Vec<CodeRow> = serde_json::from_str(json).unwrap_or_default();
    Table {
        by_code: rows
            .into_iter()
            .map(|r| (r.code, (r.text, r.text_en)))
            .collect(),
    }
}

/// status codes keyed by the status type the input carries (`status.status_type`), not by `code`
fn status_table(json: &'static str) -> Table {
    let rows: Vec<CodeRow> = serde_json::from_str(json).unwrap_or_default();
    Table {
        by_code: rows
            .into_iter()
            .filter_map(|r| Some((r.status_type?, (r.text, r.text_en))))
            .collect(),
    }
}

struct Tables {
    sni: Table,
    case_codes: Table,
    shortage: Table,
    roles: Table,
    status: Table,
    order_causes: Table,
    order_recall: Table,
    fskatt_recall: Table,
    debt_types: Table,
    enforcement_causes: Table,
    no_assets: Table,
    permits: Table,
    findings: Table,
    municipalities: Table,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        sni: table(include_str!("../texts/sni_2025.json")),
        case_codes: table(include_str!("../texts/bolagsverket_case_codes.json")),
        shortage: table(include_str!("../texts/bolagsverket_shortage_codes.json")),
        roles: table(include_str!("../texts/bolagsverket_role_codes.json")),
        status: status_table(include_str!("../texts/bolagsverket_status_codes.json")),
        order_causes: table(include_str!(
            "../texts/bolagsverket_liquidation_order_causes.json"
        )),
        order_recall: table(include_str!(
            "../texts/bolagsverket_liquidation_order_recall_causes.json"
        )),
        fskatt_recall: table(include_str!(
            "../texts/skatteverket_fskatt_recall_causes.json"
        )),
        debt_types: table(include_str!("../texts/kfm_debt_types.json")),
        enforcement_causes: table(include_str!(
            "../texts/kfm_execution_termination_causes.json"
        )),
        no_assets: table(include_str!("../texts/kfm_no_assets_reasons.json")),
        permits: table(include_str!(
            "../texts/folkhalsomyndigheten_permit_categories.json"
        )),
        findings: table(include_str!("../texts/report_findings.json")),
        municipalities: table(include_str!("../texts/scb_municipalities.json")),
    })
}

/// the industry sections (A–V): the letter, the names of its divisions (from the code table) and
/// the NACE Rev. 2.1 title (English)
pub fn sections() -> &'static [(String, String, String)] {
    #[derive(Deserialize)]
    struct Row {
        code: String,
        text: String,
        section: String,
    }
    static S: OnceLock<Vec<(String, String, String)>> = OnceLock::new();
    S.get_or_init(|| {
        let rows: Vec<Row> = serde_json::from_str(include_str!("../texts/sni_2025.json")).unwrap_or_default();
        NACE_SECTIONS
            .iter()
            .map(|(letter, en)| {
                let sv = rows
                    .iter()
                    .filter(|r| r.code.len() == 2 && r.section == *letter)
                    .map(|r| r.text.as_str())
                    .collect::<Vec<_>>()
                    .join("; ");
                ((*letter).to_owned(), sv, (*en).to_owned())
            })
            .collect()
    })
}

/// the NACE Rev. 2.1 section titles, the sections of SNI 2025
const NACE_SECTIONS: &[(&str, &str)] = &[
    ("A", "agriculture, forestry and fishing"),
    ("B", "mining and quarrying"),
    ("C", "manufacturing"),
    ("D", "electricity, gas, steam and air conditioning supply"),
    ("E", "water supply; sewerage, waste management and remediation activities"),
    ("F", "construction"),
    ("G", "wholesale and retail trade"),
    ("H", "transportation and storage"),
    ("I", "accommodation and food service activities"),
    ("J", "publishing, broadcasting, and content production and distribution activities"),
    ("K", "telecommunication, computer programming, consulting, computing infrastructure, and other information service activities"),
    ("L", "financial and insurance activities"),
    ("M", "real estate activities"),
    ("N", "professional, scientific and technical activities"),
    ("O", "administrative and support service activities"),
    ("P", "public administration and defence; compulsory social security"),
    ("Q", "education"),
    ("R", "human health and social work activities"),
    ("S", "arts, sports and recreation"),
    ("T", "other service activities"),
    ("U", "activities of households as employers; undifferentiated goods- and services-producing activities of households for own use"),
    ("V", "activities of extraterritorial organisations and bodies"),
];

const MONTHS: [(&str, &str); 12] = [
    ("januari", "January"),
    ("februari", "February"),
    ("mars", "March"),
    ("april", "April"),
    ("maj", "May"),
    ("juni", "June"),
    ("juli", "July"),
    ("augusti", "August"),
    ("september", "September"),
    ("oktober", "October"),
    ("november", "November"),
    ("december", "December"),
];

fn month_text(value: &str) -> Option<(String, Option<String>)> {
    let m: usize = value.parse().ok()?;
    let (sv, en) = MONTHS.get(m.checked_sub(1)?)?;
    Some(((*sv).to_owned(), Some((*en).to_owned())))
}

/// camelCase or snake_case keys as words: `maletFullbetalt` → `malet fullbetalt`
pub fn humanise(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut prev_lower = false;
    for c in s.chars() {
        if c == '_' || c == '-' {
            out.push(' ');
            prev_lower = false;
        } else if c.is_uppercase() && prev_lower {
            out.push(' ');
            out.extend(c.to_lowercase());
            prev_lower = false;
        } else {
            out.push(c);
            prev_lower = c.is_lowercase() || c.is_ascii_digit();
        }
    }
    out
}

fn inline(field: &str, value: &str) -> Option<(&'static str, &'static str)> {
    Some(match (field, value) {
        ("board.gap", "not_authorized") => ("Styrelsen är inte behörig", "board not authorised"),
        ("board.gap", "remark") => ("Anmärkning mot styrelsen", "remark on the board"),
        ("board.gap", "not_full") => ("Styrelsen är inte fulltalig", "board not complete"),
        ("board.gap", "missing_board_members") => {
            ("Styrelseledamöter saknas", "board members missing")
        }
        ("board.gap", "missing_ceo") => {
            ("Verkställande direktör saknas", "chief executive missing")
        }
        ("board.gap", "missing_auditor") => ("Revisor saknas", "auditor missing"),
        ("board.gap", "missing_director") => ("Föreståndare saknas", "manager missing"),
        ("board.gap", "missing_service_recipient") => (
            "Särskild delgivningsmottagare saknas",
            "special service recipient missing",
        ),
        ("board.gap", "missing_liquidator") => ("Likvidator saknas", "liquidator missing"),
        ("board.gap", "missing_company_men") => ("Bolagsmän saknas", "partners missing"),
        ("board.gap", "missing_partner") => ("Komplementär saknas", "general partner missing"),
        ("board.gap", "missing_trader") => ("Näringsidkare saknas", "trader missing"),
        ("board.gap", "vacancy") => ("Vakans bland bolagsmännen", "vacancy among the partners"),
        ("debt.case_type", "0") | ("enforcement.case_type", "0") => (
            "Allmänt mål (skatter, avgifter, böter)",
            "public claim (taxes, fees, fines)",
        ),
        ("debt.case_type", "1") | ("enforcement.case_type", "1") => {
            ("Enskilt mål (privaträttslig fordran)", "private claim")
        }
        ("enforcement.case_type", "utmatningsmal") => ("Utmätningsmål", "attachment case"),
        ("enforcement.case_type", "atertagningsmal") => ("Återtagningsmål", "repossession case"),
        ("payment_order.notification", "I") => {
            ("Insättning (ny ansökan)", "insertion (new application)")
        }
        ("payment_order.notification", "Ä") => ("Ändring", "change"),
        ("payment_order.notification", "B") => ("Borttag", "removal"),
        ("payment_order.contested", "1") => {
            ("Ansökan helt bestridd", "application fully contested")
        }
        ("payment_order.contested", "2") => {
            ("Ansökan delvis bestridd", "application partly contested")
        }
        ("payment_order.removal_cause", "0") => ("Annan orsak", "other cause"),
        ("payment_order.removal_cause", "2") => ("Målet är blockerat", "the case is blocked"),
        ("liquidation.decider", "court") => {
            ("Beslutad av tingsrätt", "decided by the district court")
        }
        ("liquidation.decider", "bolagsverket") => {
            ("Beslutad av Bolagsverket", "decided by the registry")
        }
        ("liquidation.decider", "voluntary") => (
            "Frivillig, beslutad av bolagsstämman",
            "voluntary, decided by the general meeting",
        ),
        ("case.outcome", "ceased") => ("Ärendet avslutat", "case ceased"),
        ("case.outcome", "written_off") => ("Ärendet avskrivet", "case written off"),
        ("case.outcome", "recalled") => ("Ärendet återkallat", "case recalled"),
        ("case.outcome", "refused") => ("Ansökan avslagen", "application refused"),
        ("case.outcome", "unknown") => ("Okänt utfall", "unknown outcome"),
        ("report.ruleset", "K2") => (
            "K2, förenklat regelverk för mindre företag",
            "K2, the simplified framework for smaller companies",
        ),
        ("report.ruleset", "K3") => ("K3, huvudregelverket", "K3, the main framework"),
        ("report.ruleset", "K3K") => ("K3 koncernredovisning", "K3 consolidated accounts"),
        ("report.ruleset", "K4") => ("K4, IFRS", "K4, IFRS"),
        ("report.source", "vendor") => ("Leverantörens siffror", "the data vendor's figures"),
        ("report.source", "digital") => (
            "Digital årsredovisning (iXBRL)",
            "digital annual report (iXBRL)",
        ),
        ("report.source", "scanned") => (
            "Inläst från skannad årsredovisning",
            "read from a scanned annual report",
        ),
        ("report.precision", "thousand") => ("Tusental kronor", "thousands of kronor"),
        ("report.precision", "exact") => ("Exakta kronor", "exact kronor"),
        ("dividend.type", "0") => ("Ordinarie utdelning", "ordinary dividend"),
        ("dividend.type", "1") => ("Extra utdelning", "special dividend"),
        ("name.naming_type", "0") => ("Företagsnamn", "legal name"),
        ("name.naming_type", "1") => ("Allmänt namn", "common name"),
        ("name.naming_type", "2") => ("Handelsnamn", "trading name"),
        ("name.naming_type", "3") => ("Särskilt företagsnamn", "particular business name"),
        ("name.naming_type", "4") => ("Företagsnamn på främmande språk", "foreign-language name"),
        ("name.naming_type", "5") => ("Annat namn", "other name"),
        ("address.type", "0") => ("Postadress", "mail address"),
        ("address.type", "1") => ("Besöksadress", "visiting address"),
        ("address.type", "2") => ("Post- och besöksadress", "mail and visiting address"),
        ("address.type", "3") => ("Registrerad adress", "registered address"),
        ("ownership.extent", "INTERVALL1") => (
            "Mer än 0 % men inte mer än 25 %",
            "more than 0 % but not more than 25 %",
        ),
        ("ownership.extent", "INTERVALL2") => (
            "Mer än 25 % men inte mer än 50 %",
            "more than 25 % but not more than 50 %",
        ),
        ("ownership.extent", "INTERVALL3") => (
            "Mer än 50 % men inte mer än 75 %",
            "more than 50 % but not more than 75 %",
        ),
        ("ownership.extent", "INTERVALL4") => (
            "Mer än 75 % men mindre än 100 %",
            "more than 75 % but less than 100 %",
        ),
        ("ownership.extent", "INTERVALL5") => ("100 %", "100 %"),
        ("ownership.govern", "ART10") => (
            "Kontroll genom aktier, andelar, medlemskap, avtal eller bestämmelse",
            "control through shares, units, membership, agreement or provisions",
        ),
        ("ownership.govern", "ART14") => (
            "Styrelseledamot eller motsvarande befattningshavare",
            "board member or equivalent officer",
        ),
        ("ownership.govern", "ART17") => (
            "Företräder förvaltaren av stiftelsen",
            "represents the administrator of the foundation",
        ),
        ("ownership.govern", "ART20") => (
            "Rätt att utse eller avsätta mer än hälften av styrelseledamöterna",
            "right to appoint or remove more than half of the board",
        ),
        ("ownership.govern", "ART25") => (
            "Kan få en väsentlig del av de medel som stiftelsen delar ut",
            "may receive a substantial part of the foundation's distributions",
        ),
        ("ownership.govern", "ART30") => (
            "Kontroll tillsammans med närstående",
            "control together with related parties",
        ),
        ("ownership.govern", "ART40") => (
            "Kontroll genom andra företag eller föreningar",
            "control through other companies or associations",
        ),
        ("role.auditor_type", "0") => ("Auktoriserad revisor", "certified public accountant"),
        ("role.auditor_type", "1") => ("Godkänd revisor", "approved accountant"),
        ("role.auditor_type", "2") => ("Utländsk revisor", "foreign auditor"),
        ("tax.kind", "vat") => ("Moms", "VAT"),
        ("tax.kind", "employer") => ("Arbetsgivaravgifter", "employer contributions"),
        ("tax.kind", "income") => ("Inkomstskatt", "income tax"),
        ("tax.fskatt", "F") => ("Godkänd för F-skatt", "approved for F-tax"),
        ("tax.fskatt", "FA") => (
            "Godkänd för F-skatt med villkor (FA-skatt)",
            "approved for F-tax with conditions (FA)",
        ),
        ("tax.fskatt", "none") => ("Inte godkänd för F-skatt", "not approved for F-tax"),
        ("profile.legal_form", "AB") => ("Aktiebolag", "limited company"),
        ("profile.legal_form", "HB") => ("Handelsbolag", "general partnership"),
        ("profile.legal_form", "KB") => ("Kommanditbolag", "limited partnership"),
        ("profile.legal_form", "EF") => ("Ekonomisk förening", "economic association"),
        ("profile.size_band", "unknown") => ("Okänt antal anställda", "employees unknown"),
        ("profile.size_band", "0") => ("Inga anställda", "no employees"),
        ("profile.size_band", "1-4") => ("1–4 anställda", "1–4 employees"),
        ("profile.size_band", "5-9") => ("5–9 anställda", "5–9 employees"),
        ("profile.size_band", "10-19") => ("10–19 anställda", "10–19 employees"),
        ("profile.size_band", "20-49") => ("20–49 anställda", "20–49 employees"),
        ("profile.size_band", "50-99") => ("50–99 anställda", "50–99 employees"),
        ("profile.size_band", "100-199") => ("100–199 anställda", "100–199 employees"),
        ("profile.size_band", "200-499") => ("200–499 anställda", "200–499 employees"),
        ("profile.size_band", "500+") => ("500 eller fler anställda", "500 or more employees"),
        ("interval.name", "board_gap") => (
            "Brist i styrelsen eller företrädarna",
            "board or representative gap",
        ),
        ("interval.name", "kfm_debt") => (
            "Skuld hos Kronofogden",
            "debt with the enforcement authority",
        ),
        ("interval.name", "fskatt") => ("F-skattestatus", "F-tax status"),
        ("interval.name", "vat") => ("Momsregistrering", "VAT registration"),
        ("interval.name", "employer") => ("Arbetsgivarregistrering", "employer registration"),
        ("interval.name", "liquidation") => ("Under likvidation", "in liquidation"),
        ("interval.name", "reconstruction") => {
            ("Under företagsrekonstruktion", "in company reconstruction")
        }
        ("interval.name", "liquidation_order") => (
            "Likvidationsföreläggande utan återkallelse",
            "liquidation order outstanding",
        ),
        ("interval.name", "tax_deferral") => (
            "Utestående tillfälligt anstånd",
            "outstanding temporary tax deferral",
        ),
        ("interval.name", "serving_permit") => {
            ("Gällande serveringstillstånd", "serving permit in force")
        }
        ("interval.name", "open_cases") => {
            ("Öppna ärenden hos Bolagsverket", "open registry cases")
        }
        ("sector.figure", "DebtEquityRatio") => ("Skuldsättningsgrad", "debt-to-equity ratio"),
        ("sector.figure", "EquityRatio") => ("Soliditet", "equity ratio"),
        ("sector.figure", "NetMargin") => ("Nettomarginal", "net margin"),
        ("sector.figure", "NetSalesPerEmployee") => {
            ("Nettoomsättning per anställd", "net sales per employee")
        }
        ("sector.figure", "OperatingMargin") => ("Rörelsemarginal", "operating margin"),
        ("sector.figure", "QuickRatio") => ("Kassalikviditet", "quick ratio"),
        ("sector.figure", "ReturnOnAssets") => ("Avkastning på totalt kapital", "return on assets"),
        ("sector.figure", "ReturnOnEquity") => ("Avkastning på eget kapital", "return on equity"),
        ("sector.position", "below_q1") => (
            "Under branschens första kvartil",
            "below the industry's first quartile",
        ),
        ("sector.position", "within") => (
            "Inom branschens kvartilavstånd",
            "within the industry's interquartile range",
        ),
        ("sector.position", "above_q3") => (
            "Över branschens tredje kvartil",
            "above the industry's third quartile",
        ),
        ("coverage.state", "live") => ("Källan är i drift", "source live"),
        ("coverage.state", "not_yet") => ("Källan har inte börjat än", "source not yet live"),
        ("coverage.state", "ended") => ("Källan har upphört", "source ended"),
        ("coverage.state", "undated") => ("Källan saknar startdatum", "source undated"),
        _ => return None,
    })
}

/// the text of a categorical value: the inline lists first, then the code tables, then the
/// humanised key
pub fn code_text(field: &str, value: &str) -> (String, Option<String>) {
    if let Some((sv, en)) = inline(field, value) {
        return (sv.to_owned(), Some(en.to_owned()));
    }
    let t = tables();
    let lookup = |tab: &Table| tab.by_code.get(value).cloned();
    let hit = match field {
        "industry.sni" | "profile.sni" => lookup(&t.sni),
        "case.code" => lookup(&t.case_codes),
        "report.code" => lookup(&t.shortage),
        "role.code" => lookup(&t.roles),
        "status.status_type" => lookup(&t.status),
        "liquidation_order.cause" => lookup(&t.order_causes).or_else(|| lookup(&t.order_recall)),
        "tax.reason" => lookup(&t.fskatt_recall),
        "debt.debt_type" | "enforcement.debt_type" => lookup(&t.debt_types),
        "enforcement.cause" => lookup(&t.enforcement_causes),
        "enforcement.reason" => lookup(&t.no_assets),
        "permit.category" => lookup(&t.permits),
        "report.finding" => lookup(&t.findings),
        "address.municipality" | "profile.region" => lookup(&t.municipalities),
        "profile.section" => sections()
            .iter()
            .find(|(k, _, _)| k == value)
            .map(|(_, sv, en)| (sv.clone(), Some(en.clone()))),
        "calendar.month" | "filing.fye_month" | "query.as_of_month" => month_text(value),
        "coverage.source" => risk_store::coverage::find(value).map(|c| (humanise(c.source), None)),
        "query.target" => QUERIES
            .iter()
            .find(|(k, _, _)| *k == value)
            .map(|(_, sv, en)| ((*sv).to_owned(), Some((*en).to_owned()))),
        "sector.band" => Some((
            format!("{value} anställda"),
            Some(format!("{value} employees")),
        )),
        "tax.period" => Some((
            format!("Deklarationsperiod {value}"),
            Some(format!("declaration period {value}")),
        )),
        "permit.municipality" => Some((format!("{value} kommun"), None)),
        "dividend.currency" => Some((format!("Valuta {value}"), Some(format!("currency {value}")))),
        _ => None,
    };
    hit.unwrap_or_else(|| (humanise(value), None))
}

/// whether a code's text came from a table or an inline list (false = humanised key)
pub fn has_text(field: &str, value: &str) -> bool {
    let (sv, _) = code_text(field, value);
    sv != humanise(value)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn tables_load_and_resolve() {
        assert_eq!(
            code_text("industry.sni", "69101").0,
            "Advokatbyråverksamhet"
        );
        assert!(code_text("case.code", "ORDN").0.contains("bolagsordning"));
        assert_eq!(
            code_text("status.status_type", "37").0,
            "Avförd, årsredovisning saknas"
        );
        assert_eq!(
            code_text("status.status_type", "37").1.as_deref(),
            Some("Struck off register, missing annual report")
        );
        assert_eq!(code_text("tax.reason", "517").0, "Missbruk av F-skatt");
        assert!(
            code_text("debt.debt_type", "20")
                .0
                .starts_with("Skattekontot")
        );
        assert_eq!(
            code_text("liquidation_order.cause", "11:3").0,
            "Årsredovisning saknas för perioden"
        );
        assert_eq!(
            code_text("liquidation_order.cause", "0002").0,
            "Inträffad konkurs"
        );
        assert_eq!(
            code_text("address.municipality", "0114").0,
            "Upplands Väsby"
        );
        assert_eq!(
            code_text("board.gap", "not_authorized").0,
            "Styrelsen är inte behörig"
        );
        assert_eq!(code_text("calendar.month", "12").0, "december");
        assert_eq!(
            code_text("enforcement.cause", "nyOkandOrsak").0,
            "ny okand orsak"
        );
        assert!(!has_text("enforcement.cause", "nyOkandOrsak"));
        assert!(has_text("report.finding", "CONTROL_BS"));
        assert_eq!(source_index("kronofogden"), 1);
        assert_eq!(source_index("tic"), 7);
    }

    #[test]
    fn closed_tables_have_unique_keys() {
        let mut seen = std::collections::HashSet::new();
        for k in KINDS {
            assert!(seen.insert(k.key), "duplicate kind {}", k.key);
        }
        seen.clear();
        for f in FIELDS {
            assert!(seen.insert(f.key), "duplicate field {}", f.key);
        }
        for name in risk_store::figures::FIGURES.iter().map(|f| f.name) {
            assert!(
                seen.contains(format!("figure.{name}").as_str()),
                "figure {name} has no text"
            );
        }
    }
}
