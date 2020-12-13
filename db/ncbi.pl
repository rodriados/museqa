#!/usr/bin/perl -w
# Museqa: Multiple Sequence Aligner using hybrid parallel computing.
# @file The script for downloading NCBI sequences.
# @author Rodrigo Siqueira <rodriados@gmail.com>
# @copyright 2020-present Oleg Khovayko, Rodrigo Siqueira
use strict;
use warnings;
use LWP::Simple;
use LWP::UserAgent;
use Time::Piece;

# This script downloads a list of sequences from the NCBI database via their public
# API to the NCBI Entrez system, that allows access to all Entrez databases. Although
# they generously offer this powerful tool for free, we're gently asked not to abuse
# their system by sending too many requests at a short period of time. Nonetheless,
# this script is highly inspired by those written by Oleg Khovayko and publicly
# available at https://www.ncbi.nlm.nih.gov/books/NBK25500 .
our $ncbibase = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/";
our $ncbipath = "efetch.fcgi";

# Checks whether at least one file is given to be downloaded. If no file is given,
# then we should bail out.
die "Usage: perl ncbi.pl database file [files...]\n"
    unless scalar @ARGV >= 2;

my @accnlist;
my $database = shift(@ARGV);
my @filelist = @ARGV;

my $timefmt  = localtime->strftime('%Y%m%d%H%M%S');
my $outfile  = sprintf("ncbi_%s.fasta", $timefmt);

# Iterating over the list of given input files. Here, we simply load all accession
# numbers from all files into memory.
foreach my $filename (@filelist) {
    push @accnlist, readfile($filename);
}

open my $file, ">", $outfile
    or die "could not create output file\n";

# Preparing the NCBI request parameters. We will request at most 5000 sequences
# at a time to avoid overloading the NCBI's servers with a single huge request.
while(my @current = splice @accnlist, 0, 5000) {
    my %request = (
            "db"      => $database
        ,   "id"      => join(",", @current)
        ,   "rettype" => 'fasta'
        ,   "retmode" => 'text'
        );

    download($file, \%request);
}

close $file;
print $outfile . "\n";

exit;

# Reads a file and returns all accession numbers contained within it.
# @param $1 The name of file to be processed.
# @return The list of accession numbers in file.
sub readfile
{
    my ($filename) = shift(@_);
    my (@accnlist);

    open(my $file, "<", $filename)
        or die "could not open file '$filename'\n";

    while(my $line = <$file>) {
        my @matches = $line =~ /([\w]+(?:\.[0-9]+)*)/ga;
        next unless scalar @matches > 0;
        push @accnlist, @matches;
    }

    close $file;
    return @accnlist;
}

# Downloads from the NCBI servers the requested sequences.
# @param $1 The file to which the requested sequences must be written on.
# @param $2 The configuration hash reference for requested sequences.
sub download
{
    my ($file) = shift(@_);
    my (%data) = %{shift(@_)};

    my ($url)  = $ncbibase . $ncbipath;

    my $agent  = new LWP::UserAgent;
       $agent->agent("elink/1.0" . $agent->agent);

    my $result = $agent->post($url, \%data);
    die "request returned an error\n" unless $result->code == 200;

    print $file $result->content;
}
