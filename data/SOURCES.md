# Corpus: 10 police-complaint documents in 4 formats

The practice corpus covers how complaints against and to the police work in India: registering an FIR, Zero FIR, the BNSS 2023 rules, arrest safeguards, custodial-death reporting and the Police Complaints Authorities.

**Important.** These files are **condensed summary notes**, paraphrased from the public sources listed below. They are **not** the original documents. They were compiled through a web reader that cannot download binary files, then saved in four formats so the pipeline has to handle each one. Each format includes realistic noise:

- **PDF:** multi-page files with a running header, page numbers, and headings that wrap across lines
- **HTML:** `<script>`, nav bars, footers, ad blocks, tables, `<dl>` FAQs
- **Markdown:** heading hierarchy, bold text, tables, blockquotes
- **TXT:** ALL-CAPS headings, `====` underlines, numbered steps

For a real hackathon, drop the official files into `data/raw/` and rerun `scripts/ingest.py`. The loaders are format-driven, not file-specific.

| # | File | Format | Topic | Based on (public source) |
|---|---|---|---|---|
| 01 | `01_lalita_kumari_fir_guidelines.pdf` | PDF | Mandatory FIR registration and preliminary inquiry | Supreme Court, *Lalita Kumari v. Govt. of U.P.*, (2014) 2 SCC 1, as summarised by LiveLaw, Indian Kanoon, Lexibal and LawLex |
| 02 | `02_prakash_singh_police_reform_directives.pdf` | PDF | 7 police reform directives, including Police Complaints Authorities | CHRI summary of *Prakash Singh v. Union of India* (2006) 8 SCC 1 (humanrightsinitiative.org) |
| 03 | `03_nhrc_spot_enquiry_handbook_notes.pdf` | PDF | How NHRC investigates complaints against the police | NHRC *Handbook on Spot Enquiries* (2014), nhrc.nic.in |
| 04 | `04_nhrc_custodial_death_reporting.html` | HTML | 24-hour intimation and 2-month reporting of custodial deaths | nhrc.nic.in/node/107220, node/118087; NHRC press release of 28 May 2025 |
| 05 | `05_police_complaints_authority_india.html` | HTML | Police Complaints Authorities by State | Wikipedia, "Police Complaints Authority (India)" (CC BY-SA 4.0) |
| 06 | `06_dk_basu_arrest_guidelines.html` | HTML | 11 arrest and detention safeguards; BNSS codification | *D.K. Basu v. State of West Bengal*, AIR 1997 SC 610, as summarised by LegalFly and CaseMine |
| 07 | `07_bnss_section_173_fir.md` | Markdown | Section 173 BNSS: e-FIR, Zero FIR, preliminary enquiry, remedy on refusal | Lexibal commentary; Vidhi Centre for Legal Policy |
| 08 | `08_zero_fir_sop.md` | Markdown | Zero FIR and e-Zero FIR transfer SOP | Legal Service India SOP articles; LiveLaw; MHA/PIB coverage of e-Zero FIR |
| 09 | `09_first_information_report_basics.txt` | TXT | FIR basics and what to do if the police refuse | Wikipedia, "First information report" (CC BY-SA 4.0) |
| 10 | `10_lok_sabha_q2277_police_complaints_authority.txt` | TXT | Government answer on PCA implementation (2012) | Lok Sabha Unstarred Q. 2277, eparlib.nic.in |

This corpus is for a learning project. It is **not legal advice**.
