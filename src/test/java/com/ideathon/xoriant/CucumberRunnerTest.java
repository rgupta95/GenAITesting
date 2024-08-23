package com.ideathon.xoriant;


import io.cucumber.junit.CucumberOptions;
import net.serenitybdd.cucumber.CucumberWithSerenity;
import org.junit.runner.RunWith;

@RunWith(CucumberWithSerenity.class)
@CucumberOptions(
        plugin={"pretty"}   ,
        features = {"src/test/resources/features/GenAIUseCases.feature"},
        glue ="com.ideathon.xoriant.stepdefinitions",
        monochrome = true
       // tags="@test"
)
public class CucumberRunnerTest {



}
