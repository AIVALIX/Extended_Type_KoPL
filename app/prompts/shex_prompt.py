shex_prompt = """
PREFIX ex: <http://example.org/ontology/>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
START = @<CreativeWork>
<CreativeWork> {
    ex:starred_actors   @<Person> * ;
    ex:has_imdb_votes   @<Number> * ;
    ex:directed_by      (@<Person> OR @<Organization>) + ;
    ex:written_by       (@<Person> OR @<Organization>) + ;
    ex:in_language      (@<Language> OR @<Text>) * ;
    ex:has_genre        @<Text> * ;
    ex:release_year     @<Date> * ;
    ex:has_tags         @<Text> * ;
    ex:has_imdb_rating  @<Number> * ;
}
<Person> {
    a [schema:Person] ;
    ^ex:starred_actors  @<CreativeWork> * ;
    ^ex:directed_by     @<CreativeWork> * ;
    ^ex:written_by      @<CreativeWork> *
}
<Organization> {
    a [schema:Organization] ;
    ^ex:directed_by     @<CreativeWork> * ;
    ^ex:written_by      @<CreativeWork> *
}
<Language> {
    a [schema:Language] ;
    ^ex:in_language     @<CreativeWork> *
}
<Number>{
    a [schema:Number] ;
    ^has_imdb_rating    @<CreativeWork> * ;
    ^ex:has_imdb_votes   @<CreativeWork> *
}
<Text>{
    a [schema:Text] ;
    ^ex:has_tags        @<CreativeWork> * ;
    ^ex:has_genre       @<CreativeWork> * ;
    ^ex:in_language     @<CreativeWork> *
}
<Date>{
    a [schema:Date] ;
    ^ex:release_year    @<CreativeWork> * ;
}   
"""
