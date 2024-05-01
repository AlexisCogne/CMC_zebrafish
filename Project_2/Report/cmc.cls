\ProvidesClass{cmc}

\LoadClass[11pt,onecolumn,a4paper,oneside,final]{article}

% Fundamental packages
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[american]{babel}
\usepackage[hidelinks]{hyperref}
\usepackage{soul}

% Mathematics packages
\usepackage{array}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{mathtools}

% Figures packages
\usepackage{graphicx}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{subcaption}
\usepackage[font={it}]{caption}

%font-codes package
\usepackage{listings}
\usepackage{color}
\usepackage{xcolor}
\usepackage{verbatim}

% Paragraph spacing
\usepackage[parfill]{parskip}

% Format
\usepackage{fancyhdr}
\usepackage{multicol}

% Paths
\usepackage{url}
\usepackage[space]{grffile}

% Paper size and Layout
\usepackage[
dvips,
a4paper,
top=1cm,
bottom=2cm,
left=2cm,
right=2cm,
headheight=25.34557pt,
headsep=10mm,
marginparwidth=2cm,
marginparsep=4mm,
heightrounded,
includehead
]{geometry}

%---- Tables ----%
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{multicol}

% ---- Title ----%
\title{Computational Motor Control - 2024}

% ----newcommand----%
\newcommand{\corr}[1]{\textcolor{red}{#1}}
\newcommand{\fileref}[1]{\textcolor{red}{\texttt{#1}}}
