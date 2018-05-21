

use strict;


my $ansfile = shift(@ARGV);
my $goldfile = shift(@ARGV);
my $type =  shift(@ARGV);
my $scoretype = shift(@ARGV);
my $verbose = shift(@ARGV);


my %ambrit = {
'molt',  'moult',
'scrutinize', 'scrutinise',
'organize', 'organise',
'utilize', 'utilise',
'realize', 'realise',
'color', 'colour', 
'actualization', 'actualisation',
'authorize', 'authorise',
'demeanor', 'demeanour',
'rigor', 'rigour',
'unspoiled' ,'unspoilt'
};

my $goldpath = "\."; # set this to path with MRsamples and mwids 
               # if you want further analysis for NMWT NMWS RAND and MAN
 top();


sub top {
if (!$ansfile || !$goldfile || ($ansfile =~ /-(t|v)$/) || ($goldfile =~ /-(t|v)/)) {
  print "score.pl usage: systemfile goldfile [-t best|oot|mw] [-v]\n";
  undef $scoretype;
}

if (!$type) {
  $scoretype = 'best';
}
elsif (($type eq '-v') && !$scoretype && !$verbose) {
  $scoretype = 'best';
  $verbose = '-v';
}
elsif ($type && ($type ne '-t')) {
  print "score.pl usage: systemfile goldfile [-t best|oot|mw] [-v]\n";
  undef $scoretype;
}

my $subana; # NMWT, only do ids without MW detected in target,
            # NMWS only using single word substitutes from GS
            # MAN only MAN sample 
            # RAND only RAND sample
my $E = 1; # exclude, for NMWT, all or MAN
# 0 for RAND

if ($scoretype) {
  if ($scoretype eq 'best') {
    scorebest($goldfile,$ansfile,$subana,$E); # supplying best answers
 #   $subana = 'NMWT';
 #   print "\nFURTHER ANALYSIS Scoring without items identified as MWs\n";
 #   scorebest($goldfile,$ansfile,$subana,$E); 
 #   $subana = 'NMWS';
 #   print  "\nFURTHER ANALYSIS Scoring using only single word substitutes\n";
 #   scorebest($goldfile,$ansfile,$subana,$E); 
 #   $subana = 'MAN';
 #   print  "\nFURTHER ANALYSIS Scoring  items selected randomly\n";
 #   scorebest($goldfile,$ansfile,$subana,$E); 
 #    $subana = 'RAND';
 #   $E = 0;
 #   print "\nFURTHER ANALYSIS Scoring  items selected manually\n";
 #   scorebest($goldfile,$ansfile,$subana,$E); 
  }
  elsif ($scoretype eq 'oot') {
    scoreOOT($goldfile,$ansfile,$subana,$E); #supplying up to 10 answers
 #   $subana = 'NMWT';
 #   print  "\nFURTHER ANALYSIS Scoring without items identified as MWs\n";
 #   scoreOOT($goldfile,$ansfile,$subana,$E); #supplying up to 10 answers
 #   $subana = 'NMWS';
 #   print  "\nFURTHER ANALYSIS Scoring using only single word substitutes\n";
 #   scoreOOT($goldfile,$ansfile,$subana,$E); #supplying up to 10 answers
 #   $subana = 'MAN';
 #   print  "\nFURTHER ANALYSIS Scoring  items selected randomly\n";
 #   scoreOOT($goldfile,$ansfile,$subana,$E); #supplying up to 10 answers
 #   $subana = 'RAND';
 #   $E = 0;
 #   print  "\n FURTHER ANALYSIS Scoring just items selected manually\n";
 #   scoreOOT($goldfile,$ansfile,$subana,$E); #supplying up to 10 answers
  }
  elsif ($scoretype eq 'mw') {
    scoreMW($goldfile,$ansfile);
  }
  elsif ($scoretype) {
    print "score.pl usage: systemfile goldfile [-t best|oot|mw] [-v]\n";
  }
}

}

sub scorebest {
  my($goldfile,$ansfile,$subana,$E) = @_;
  my(%idws,%idres,%idmodes);
  my($line,$id,$wpos,$res,@res,$numguesses,$hu,$totmodatt,$besteqmode);
  my($totitems,$idcorr,$corr,$precision,$recall,$totmodes,$itemsattempted);
  my($sub,%norms,%done,$score,$posthoc);
  my $dp = 2;
  my $lcnt = 0;
  ($totitems,$totmodes) = readgoldfile($goldfile,\%idws,\%idres,\%idmodes,$subana,$E);
  open(SYS,$ansfile);
  while ($line = <SYS>) {
    $lcnt++;    
    undef $idcorr;
    undef %norms;
    if ($line =~ /([\w.]+) (\S+) \:\: (.*)/) {
      $id = $2;
      $wpos = $1;
      $res = $3;      
      $hu = sumvalues(\%{$idres{$id}});
      normalisevalues(\%{$idres{$id}},\%norms,$hu);
      if ($hu && !$done{$id}) { # duplicates in system file
	$done{$id} = 1;
	if ($res =~ /\S/) {
	    @res = split(';',$res); # can't do on spaces because of MWEs
	    @res = fixmisc($subana,@res); # hypens, american british, apost
	    $numguesses = $#res + 1;
	    if ($numguesses) {
		$itemsattempted++;
	    }

	}
	else {
	   # print "$id $res DEBUG\n";
	}
	if ($idmodes{$id} && $numguesses) {
	  $totmodatt++;
	  if ($idmodes{$id} eq $res[0]) {
	    $besteqmode++;
	    if ($verbose) {
	      print "$wpos Item $id mode '$idmodes{$id}' : system '$res[0]'  correct\n";
	    }
	  }
	  elsif ($verbose) {
	      print "$wpos Item $id mode '$idmodes{$id}' : system '$res[0]' wrong\n";
	    }
	}
	foreach $sub (@res) {
	  $idcorr += $norms{$sub}; 
      }        
	  if ($idcorr) {
	    $score = (($idcorr / $numguesses)); 
	    $corr += $score;
	  }
	if ($verbose && $numguesses) {
	  print "$wpos Item $id credit $idcorr guesses $numguesses human responses $hu: score is $idcorr\n";
	}

    }
      else {
	 # print "not attempted $line\n"; # debugging items not included
      }
    } # item
    elsif ($line =~ /\S/) {
       print "Error in $ansfile on line $lcnt\n";
    }    
}
  $precision = $corr / $itemsattempted;
  $precision = myround($precision,$dp);
  $recall = $corr / $totitems;
  $recall = myround($recall,$dp);
  print "Total = $totitems, attempted = $itemsattempted\n";
  print "precision = $precision, recall = $recall\n";
  $precision = $besteqmode / $totmodatt; # where there was a mode and
                         # system had an answer
  $precision = myround($precision,$dp);
  $recall = $besteqmode / $totmodes;
  $recall = myround($recall,$dp);
  print "Total with mode $totmodes attempted $totmodatt\n";
  print "Mode precision = $precision, Mode recall = $recall\n";
  close(SYS);
}

sub scoreOOT {
  my($goldfile,$ansfile,$subana,$E) = @_;
  my(%idws,%idres,%idmodes);
  my($line,$id,$wpos,$res,@res,$numguesses,$hu,$totmodatt,$foundmode);
  my($totitems,$idcorr,$corr,$precision,$recall,$totmodes,$itemsattempted);
  my($sub,%norms,%done,$score);
  my $dp = 2;
  my $lcnt = 0;
  my $dupflag = 0;
  ($totitems,$totmodes) = readgoldfile($goldfile,\%idws,\%idres,\%idmodes,$subana,$E);
  # read into arrays like read agr
  open(SYS,$ansfile);
  while ($line = <SYS>) {
    $lcnt++;
    undef $idcorr;
    undef %norms;
    if ($line =~ /([\w.]+) (\S+) \:\:\: (.*)/) {
      $id = $2;
      $wpos = $1;
      $res = $3;
      $hu = sumvalues(\%{$idres{$id}});
      normalisevalues(\%{$idres{$id}},\%norms,$hu);
      if ($hu && !$done{$id}) {
	$done{$id} = 1;
	if ($res =~ /\S/) {
	  @res = split(';',$res); # can't do on spaces because of MWEs
	  @res = fixmisc($subana,@res); # hypens, american british, apost
	  if ($#res > -1) {
	      $itemsattempted++;
	  }
	  if (duplicates_p(@res)) {
	      $dupflag++;
	      if ($verbose) {
		  print "WARNING duplicate at $wpos $id responses @res\n";
	      }
	  }
	}
	else {
	#    print "$id $res\n";
	}
	if ($#res > 9) {
	  @res = @res[0..9];
	  if ($verbose) {
	    print "$wpos Item $id exceeded 10 guesses\n";
	  }
	}
	if ($idmodes{$id} && ($#res > -1)) {
	  $totmodatt++;
	  if (strmember($idmodes{$id},@res)) {
	    $foundmode++; # mode is in guesses	    
	    if ($verbose) {
	      print "$wpos Item $id mode '$idmodes{$id}'  found in guesses\n";
	    }
	  }
	  elsif ($verbose) {
	      print "$wpos Item $id mode '$idmodes{$id}'  not found\n";
	    }
	}
	foreach $sub (@res) {
	  $idcorr += $norms{$sub}; 
	}      
	if ($idcorr) {
	  $corr += $idcorr; 
	}	
	if ($verbose &&  ($#res > -1)) {
	  print "$wpos Item $id credit $idcorr human responses $hu: score is $idcorr\n";
	}
      } # if $hu, humans said something
    } # item
    elsif ($line =~ /\S/) { 
      print "Error in $ansfile on line $lcnt\n";
    }    
  }
  if ($dupflag) {
      print "WARNING OOT file contains duplicates on $dupflag lines\n";
  }
  $precision = $corr / $itemsattempted;
  $precision = myround($precision,$dp);
  $recall = $corr / $totitems;
  $recall = myround($recall,$dp);
  print "Total = $totitems, attempted = $itemsattempted\n";
  print "precision = $precision, recall = $recall\n";
  $precision = $foundmode / $totmodatt; # where there was a mode and
                         # system had an answer
  $precision = myround($precision,$dp);
  $recall = $foundmode / $totmodes;
  $recall = myround($recall,$dp);
  print "Total with mode $totmodes attempted $totmodatt\n";
  print "precision = $precision, recall = $recall\n";
  close(SYS);
}

# will take MW as mode lemmatised
# score for identifying MW on this line - precision (/attempts) 
# recall (/ actual MW)
# score for guessing correct MW, assuming lemmatised mode - min 2 verdicts

sub scoreMW {
  my($goldfile,$ansfile) = @_;
  my(%idmodes);
  my($line,$id,$wpos,$res,$sysmwtot,$corrmode,$totmodes);
  my($totitems,$idcorr,$corr,$precision,$recall,$totmodes,$sysmwatt);
  my($sub,%norms,%done);
  my $dp = 2;
  my $lcnt = 0;
  $totmodes = readmwfile($goldfile,\%idmodes); # recall denom
  open(SYS,$ansfile);
  while ($line = <SYS>) {
    $lcnt++;    
    undef $idcorr;
    undef %norms;
    if ($line =~ /([\w.]+)\s+(\d+)\s*\:\:\s*(.*)\s*$/) {
      $id = $2;
      $wpos = $1;
      $res = $3;      
      if (!$done{$id}) {
	$done{$id} = 1;
	if ($res =~ /\S/) {
	  $sysmwtot++; # prec denom
	}
	if ($idmodes{$id}) {
	  $sysmwatt++; # how many modes did system say were MWs
	  if ($idmodes{$id} eq $res) {
	    $corrmode++; # did system get MW correct?
	  } 
	  if ($verbose) {
	    print "$wpos $id human mode is $idmodes{$id} system $res\n";
	  }
	}
	elsif ($verbose) {
	  print "$wpos $id No MW found by annotators, system $res\n";
	}
	
      }
    }
    elsif ($line =~ /\S/) {
      print "Error in $ansfile on line $lcnt\n";
    }
  }
# is there a MW
  if ($sysmwtot) {
    $precision = $sysmwatt/ $sysmwtot;
    $precision = myround($precision,$dp);
  }
  $recall = $sysmwatt / $totmodes;
  $recall = myround($recall,$dp);
  print "Total MWs in GS = $totmodes, System found $sysmwtot of which $sysmwatt were genuine\n";
  print "Detection precision = $precision, recall = $recall\n";
  if ($sysmwtot) {
    $precision = $corrmode / $sysmwtot; # where systems ans matched GS,
                         # and system had an answer
    $precision = myround($precision,$dp);
  }
  $recall = $corrmode / $totmodes;
  $recall = myround($recall,$dp);
  print "Number that matched GS\n";
  print "Identification precision = $precision, recall = $recall\n";
  close(SYS);

}


sub readmwfile {
  my($gsfile,$modes) = @_;
  my($line,$id,$wpos,$rest,@res,$mw,$num,$mode,$modenum,$i,$totitems,$totmodes);
  my($res,$first,$ms);
  my $lcnt = 0;
  open(GS,"$gsfile");
  while ($line = <GS>) {
    $lcnt++;
    if ($line =~ /([\w.]+)\s+(\d+)\s*\:\: (.*)/) {
      $id = $2;
      $wpos = $1;
      $rest = $3;
      undef $mode;
      undef $modenum;
      undef $ms;
      undef $num;
      @res = split(';',$rest);     
      $first = $res[0];
      if ($first =~ /[\w\'-\s]+ (\d+)/) {
	  $num = $1;
      }
      if ($num > 1) { # for mws want 2 humans to have same 
                    # response (though after lemmatising
	$totitems++;
	foreach $res (@res) {
	  if ($res =~ /(\w[\w-\'\s]+) (\d+)/) {
	    $mw = $1;
	    $num = $2;
	    $mw =~ s/'//;# remove apostrophe, should only be one
	  if ((!$mode) && ($num > 1)) { 
                    # also cond below will take care of those of 1
	            # though we will only take ids where at least 2 responses
	    $mode = $mw;	  
	    $modenum = $num;
	    $$modes{$id} = $mode;	 
	    $totmodes++;
	  }
	  elsif (!$ms && $mode && ($num == $modenum)) { # mode found was not the most freq	  
	    delete $$modes{$id};	 
	    $totmodes--;
	    $ms = 1; # so we don't do this twice for 1 id
	  }
	  }	 
	}     
    }
  }
    elsif ($line =~ /\S/) {
       print "Error in $gsfile on line $lcnt\n";
    }
}
  close(GS);
 # printinfile($modes,"mwids"); # just do once to create for scorebest and oot
  return $totmodes;
}



sub fixmisc {
  my($sa,@ans) = @_;
  my ($item,@result,$debug);
  foreach $item (@ans) {     
    if ($item =~ /^non(\s|-)(.*)$/) {
      $item = "non$2";
    }
    $item =~ s/-/ /g; 
    $item =~ s/'//; # remove one apostrophe
    if ($ambrit{$item}) {
      $item = $ambrit{$item};
    }
   if (($sa eq 'NMWS') && ($item =~ /\S \S/)) {# don't include MW subs 
       $debug .= "$item;";
   }
   else  {
    push(@result,$item);
   }
}
  return @result;
}

# gives ids (mws) or wpos to exclude from scoring
sub subset {
    my($subana) = @_;
    my(%result);
    if ($subana eq 'NMWT') {
	%result = getnmw("$goldpath/mwids");
    }
    if ($subana =~  /RAND|MAN/) {
	%result = getman("$goldpath/MRsamples");
    }
    return %result;

}
sub getman {
    my($file) = @_;
    my($line,%result);
    open(SUB,"$file");
    while ($line = <SUB>) {
	if ($line =~ /(\w+);(\w)\.man/) {	    
	    $result{"$1\.$2"} = 1;
	}
    }
    close(SUB);
    return %result;
}


sub getnmw {
    my($file) = @_;
    my($line,%result);
    open(SUB,"$file");
    while ($line = <SUB>) {
	if ($line =~ /^(\d+) /) {
	    $result{$1} = 1;
	}
    }
    close(SUB);
    return %result;
}

# set E as 1 if want to exclude  subset, or evaluate all
# set E as 0 if want to only use subset (for MAN)
sub readgoldfile {
  my($gsfile,$idwarr,$resarr,$modes,$subana,$E) = @_;
  my($line,$id,$wpos,$rest,@res,$sub,$num,$mode,$modenum,$i,$totitems,$totmodes);
  my($res,$ms,$first,%exclude);
  %exclude = subset($subana);
  open(GS,"$gsfile");
  while ($line = <GS>) {
    if ($line =~ /([\w.]+) (\S+) \:\: (.*)/) {
      $id = $2;
      $wpos = $1;
      $rest = $3;
      undef $mode;
      undef $modenum;
      undef $ms;
      undef $num;
      # !$E && $exclude{$wpos} MAN - use just these
      # $E and nothing in %exclude - use all
      # $E and !$exclude{$id} - not one of MWs we are ignoring
      #  $E and !$exclude{$wpos} - not one of manuals we are ignoring (i.e. RAND set)
      if(($E && (!$exclude{$id} && !$exclude{$wpos})) || (!$E && $exclude{$wpos})) { # mw ids, man rand are wpos
	  @res = split(';',$rest);
	  @res = removeifpatt('pn',@res); # 
	  if ($subana eq 'NMWS') { # not MWs as substitutes
	      @res = removeifpatt('\s\S+\s+\d+',@res);
	  }
	  $first = $res[0];
	  if ($first =~ /[\w\'-\s]+ (\d+)/) {
	      $num = $1;
	  }
	  if (($#res > 0) || ($num > 1)) { # i.e. 2 or morenon nil and non pn (proper noun)
	      $totitems++;
	      $$idwarr{$id} = $wpos;
	      foreach $res (@res) {
		  if ($res =~ /(\w[\w\'-\s]+) (\d+)/) {
		      $sub = $1;
		      $num = $2;
		      $sub =~ s/'//;# remove apostrophe, should only be one
		      if (!$mode) { # cond below will take care of those of 1
			  $mode = $sub;	  
			  $modenum = $num;
			  $$modes{$id} = $mode;	 
			  $totmodes++;
		      }
		      elsif (!$ms && $mode && ($num == $modenum)) { # mode found was not the most freq	  
			  delete $$modes{$id};	 
			  $totmodes--;
			  $ms = 1; # so we don't do this twice for 1 id
		      }
		      $$resarr{$id}{$sub} = $num;
		  }	 
	      }      
      }
      else {
	  #print "can't use: $line\n"; # debugging
      }

  }# is part of group we are scoring
      else {
	  #print "not in group: $line\n"; # debugging
      }
  } # matches line
  else {
      #print "GS mismatch: $line\n"; # debugging
  }
} # while
  close(GS);

  return ($totitems,$totmodes);
}



###  utilities
sub removeifpatt {
	my($patt,@list) = @_;
	my($i,@result);
	foreach $i (@list) {
		if ($i !~ /$patt/) {
			push(@result,$i);
		}
	}	
	return @result;
}

# rounds by $db decimal places
sub myround {
    my($number,$dp) = @_;
	my($mult,$result,$dec,$len,$dpo,$diff);
    $number *= 100; # to give percentages
	$dpo = '0' x $dp;
	$mult = 1 . $dpo;
	$number = $number * $mult;
    	$result =  int($number + .5);
	$result = $result /  $mult;
	if ($result =~ /\.(\d+)/) {
		$dec = $1;
		$len = length($dec);
		$diff = $dp - $len;
		while ($diff) {
			$result .= '0';
			$diff--;
		}
	}
	else {
		$result .= ".$dpo";
	}
	return $result;
}


sub sumvalues {
    my($array) = @_;
    my($key,$val,$result);
    while (($key,$val) = each %$array) {
        $result += $val;
    }
    return $result;
}


# normalise by sum, and account for hyphens  in GS
# so if humans put hyphens, then could be spaces or hyphens
# but if humans didn't and systems do systems may be found incorrect
sub normalisevalues {
  my($source,$target,$sum) = @_;
  my($key,$val,$key2,$result);
  while (($key,$val) = each %$source) {
        $$target{$key} = $val / $sum;
	if ($key =~ /-/) {
	  $key2 = $key;
	  $key2 =~ s/-/ /g;
	  $$target{$key2} = $val / $sum;
	}
    }
}

sub strhypmember {
  my($item,@list) = @_;
    my($index);
    for ($index = 0; $index <= $#list; $index++) {
        if (myequal($list[$index],$item)) { # item is GS
            return $index + 1;
            last;
        }
    }
}

sub myequal {
  my($item,$item2) = @_;
  if ($item eq $item2) {
    return 1;
  }
  elsif ($item2 =~ /-/) {
    $item2 =~ s/-/ /g; # get rid of hyphens in GS
    if ($item eq $item2) {
      return 1;
    }
  }
  else {
    return 0;
  }
}

sub strmember {
    my($item,@list) = @_;
    my($index);
    for ($index = 0; $index <= $#list; $index++) {
        if ($list[$index] eq $item) {
            return $index + 1;
            last;
        }
    }
}




sub removeall {
        my($item,@list) = @_;
        my($i,@result);
        foreach $i (@list) {
                if ($i ne $item) {
                        push(@result,$i);
                }
        }
        return @result;
}




sub printinfile {
	my($array,$file) = @_;
	my($key,@keys);
	open(PSBVFILE,">>$file");
	@keys = keys %$array;
	foreach $key (@keys) {
	    print PSBVFILE "$key  $$array{$key}\n\n"
	}
	close(PSBVFILE);

}

sub duplicates_p {
	my(@list) = @_;
	my($i);
	for($i = 0; $i <= $#list; $i++) {
        	if (strmember($list[$i],@list[$i+1..$#list])) {
            		return 1;
        	} #if
    	} # for
    	return 0;
}
